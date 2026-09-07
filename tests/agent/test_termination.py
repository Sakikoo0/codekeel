import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from codekeel.agent import Agent, RunStatus
from codekeel.models import FakeModel, Message, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from codekeel.runtime import BudgetLimits
from codekeel.runtime.policy import ActionPolicy, Risk
from codekeel.tools import ToolContext, ToolRegistry


class FakeWorkspace:
    """Workspace double; budget tests deliberately use no external execution."""


@dataclass
class CountingTool:
    name: str = "tick"
    description: str = "Record one deterministic tool call."
    calls: list[dict[str, Any]] = field(default_factory=list)

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={"type": "object", "additionalProperties": False},
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(content="tick")


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class ClockAdvancingFakeModel(FakeModel):
    def __init__(self, responses: list[ModelResponse], clock: FakeClock, *, advance: float) -> None:
        super().__init__(responses)
        self.clock = clock
        self.advance = advance

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        response = await super().complete(messages, tools=tools)
        self.clock.advance(self.advance)
        return response


class CancellingFakeModel(FakeModel):
    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        raise asyncio.CancelledError


def _tool_response(usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id="call", name="tick", arguments={})],
        usage=usage or Usage(),
    )


def _isolated_budgets(**limit) -> BudgetLimits:
    values = {
        "max_steps": None,
        "max_model_calls": None,
        "max_tool_calls": None,
        "max_wall_time": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "max_cost": None,
    }
    values.update(limit)
    return BudgetLimits(**values)


async def test_agent_completes_within_budgets() -> None:
    tool = CountingTool()
    agent = Agent(
        model=FakeModel(
            [
                _tool_response(Usage(input_tokens=2, output_tokens=1, cost=0.1)),
                ModelResponse(content="done"),
            ]
        ),
        workspace=FakeWorkspace(),
        policy=ActionPolicy(tool_risks={"tick": Risk.LOW}), tool_registry=ToolRegistry([tool]),
        budgets=_isolated_budgets(max_steps=3),
    )

    state = await agent.run("finish")

    assert state.status is RunStatus.COMPLETED
    assert (state.steps, state.model_calls, state.tool_calls) == (2, 2, 1)
    assert state.usage == Usage(input_tokens=2, output_tokens=1, cost=0.1)
    assert tool.calls == [{}]


@pytest.mark.parametrize("limit", [{"max_steps": 1}, {"max_model_calls": 1}, {"max_cost": 0.5}])
async def test_completion_on_the_last_allowed_model_call_wins(limit) -> None:
    agent = Agent(
        model=FakeModel([ModelResponse(content="done", usage=Usage(cost=0.5))]),
        workspace=FakeWorkspace(),
        budgets=_isolated_budgets(**limit),
    )

    state = await agent.run("finish at the boundary")

    assert state.status is RunStatus.COMPLETED
    assert (state.steps, state.model_calls, state.tool_calls) == (1, 1, 0)


@pytest.mark.parametrize(
    ("budgets", "usage", "expected_status"),
    [
        (_isolated_budgets(max_steps=2), Usage(), RunStatus.MAX_STEPS),
        (_isolated_budgets(max_model_calls=2), Usage(), RunStatus.MAX_STEPS),
        (_isolated_budgets(max_tool_calls=2), Usage(), RunStatus.MAX_STEPS),
        (_isolated_budgets(max_input_tokens=5), Usage(input_tokens=3), RunStatus.MAX_TOKENS),
        (_isolated_budgets(max_output_tokens=5), Usage(output_tokens=3), RunStatus.MAX_TOKENS),
        (_isolated_budgets(max_cost=0.5), Usage(cost=0.3), RunStatus.MAX_COST),
    ],
)
async def test_repeating_tool_calls_stop_at_each_budget(budgets, usage, expected_status) -> None:
    tool = CountingTool()
    agent = Agent(
        model=FakeModel([_tool_response(usage) for _ in range(4)]),
        workspace=FakeWorkspace(),
        policy=ActionPolicy(tool_risks={"tick": Risk.LOW}), tool_registry=ToolRegistry([tool]),
        budgets=budgets,
    )

    state = await agent.run("keep calling tools")

    assert state.status is expected_status
    assert state.steps == 2
    assert state.model_calls == 2
    assert state.tool_calls == 2
    assert len(tool.calls) == 2


async def test_wall_time_uses_injected_clock_without_sleep() -> None:
    clock = FakeClock(10.0)
    tool = CountingTool()
    model = ClockAdvancingFakeModel([_tool_response()], clock, advance=2.0)
    agent = Agent(
        model=model,
        workspace=FakeWorkspace(),
        policy=ActionPolicy(tool_risks={"tick": Risk.LOW}), tool_registry=ToolRegistry([tool]),
        budgets=_isolated_budgets(max_wall_time=1.0),
        clock=clock,
    )

    state = await agent.run("time out")

    assert state.status is RunStatus.TIMEOUT
    assert (state.steps, state.model_calls, state.tool_calls) == (1, 1, 0)
    assert tool.calls == []


async def test_external_task_cancellation_sets_cancelled_status() -> None:
    agent = Agent(
        model=CancellingFakeModel([]),
        workspace=FakeWorkspace(),
        budgets=_isolated_budgets(max_steps=1),
    )

    with pytest.raises(asyncio.CancelledError):
        await agent.run("cancel")

    assert agent.state.status is RunStatus.CANCELLED


def test_run_status_contains_current_state_set() -> None:
    assert list(RunStatus) == [
        RunStatus.IDLE,
        RunStatus.RUNNING,
        RunStatus.WAITING_FOR_APPROVAL,
        RunStatus.COMPLETED,
        RunStatus.FAILED,
        RunStatus.VERIFICATION_FAILED,
        RunStatus.MAX_STEPS,
        RunStatus.MAX_COST,
        RunStatus.MAX_TOKENS,
        RunStatus.TIMEOUT,
        RunStatus.CANCELLED,
    ]