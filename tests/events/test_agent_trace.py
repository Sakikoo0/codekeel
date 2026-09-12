import asyncio
from unittest.mock import AsyncMock

import pytest

from codekeel.agent import Agent, AgentProtocolError, RunStatus
from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.store import EventStoreError, MemoryEventStore
from codekeel.models import FakeModel, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from codekeel.runtime import BudgetLimits
from codekeel.runtime.policy import ActionPolicy, Risk
from codekeel.tools import ToolRegistry


class FakeWorkspace:
    """No external filesystem or commands are used by these core tests."""


class RecordingTool:
    name = "record"
    description = "Deterministic tool"

    def __init__(self, result=None, error=None):
        self.result = result or ToolResult(content="observation")
        self.error = error
        self.calls = []

    def definition(self):
        return ToolDefinition(name=self.name, description=self.description, parameters={"type": "object"})

    async def execute(self, arguments, context):
        self.calls.append(context.run_id)
        if self.error:
            raise self.error
        return self.result


def response():
    return ModelResponse(
        tool_calls=[ToolCall(id="call-1", name="record", arguments={"key": "value"})],
        usage=Usage(input_tokens=4, output_tokens=2, cost=0.2),
    )


def trace(agent):
    return agent.event_store.read(agent.run_id).events


async def test_deterministic_trajectory_order_payloads_and_run_isolation(tmp_path):
    tool = RecordingTool()
    agent = Agent(
        FakeModel([response(), ModelResponse(content="done"), ModelResponse(content="again")]),
        FakeWorkspace(), policy=ActionPolicy(tool_risks={"record": Risk.LOW}),
                  tool_registry=ToolRegistry([tool]), event_store=JsonlEventStore(tmp_path),
    )
    state = await agent.run("task")
    first_run = agent.run_id
    events = trace(agent)
    assert [event.type for event in events] == [
        "RunStarted", "BudgetUpdated", "ModelRequested", "ModelResponded", "BudgetUpdated",
        "BudgetUpdated", "ToolCalled", "ToolCompleted", "BudgetUpdated", "ModelRequested",
        "ModelResponded", "BudgetUpdated", "RunFinished",
    ]
    assert [event.sequence for event in events] == list(range(1, 14))
    assert len({event.event_id for event in events}) == 13
    assert {event.run_id for event in events} == {first_run}
    assert tool.calls == [first_run]
    assert events[0].payload.messages == state.messages[:2]
    assert events[2].payload.messages[:-1] == state.messages[:2]
    assert events[9].payload.messages[:-1] == state.messages[:4]
    assert events[3].payload.response == response()
    assert events[6].payload.call == response().tool_calls[0]
    assert events[7].payload.tool_call_id == "call-1"
    assert events[7].payload.result == tool.result
    assert events[-1].payload.status == "completed"
    assert events[-1].payload.usage == state.usage
    assert events[-1].payload.model_calls == 2
    assert events[-1].payload.tool_calls == 1
    await agent.run("second task")
    assert agent.run_id != first_run
    assert trace(agent)[0].sequence == 1
    assert agent.event_store.read(first_run).events == events


async def test_recoverable_tool_failure_is_recorded_and_agent_can_finish():
    tool = RecordingTool(ToolResult(content="not found", is_error=True))
    agent = Agent(FakeModel([response(), ModelResponse(content="done")]), FakeWorkspace(),
                  policy=ActionPolicy(tool_risks={"record": Risk.LOW}), tool_registry=ToolRegistry([tool]))
    await agent.run("task")
    failed = next(event for event in trace(agent) if event.type == "ToolFailed")
    assert failed.payload.result == tool.result
    assert failed.payload.tool_call_id == "call-1"
    assert trace(agent)[-1].type == "RunFinished"
    assert not any(event.type == "ToolCompleted" for event in trace(agent))


@pytest.mark.parametrize("error", [RuntimeError("sensitive error details"), asyncio.CancelledError(), TimeoutError()])
async def test_tool_exception_timeout_and_cancellation_have_terminal_events(error):
    agent = Agent(FakeModel([response()]), FakeWorkspace(),
                  policy=ActionPolicy(tool_risks={"record": Risk.LOW}),
                  tool_registry=ToolRegistry([RecordingTool(error=error)]))
    if isinstance(error, TimeoutError):
        await agent.run("task")
        assert agent.state.status is RunStatus.TIMEOUT
    else:
        with pytest.raises(type(error)):
            await agent.run("task")
    events = trace(agent)
    failed = next(event for event in events if event.type == "ToolFailed")
    assert failed.payload.error_type == type(error).__name__
    assert events[-1].type == ("RunFailed" if isinstance(error, RuntimeError) else "RunFinished")
    assert "sensitive error details" not in "".join(event.model_dump_json() for event in events)


@pytest.mark.parametrize("error", [RuntimeError("private"), asyncio.CancelledError(), TimeoutError()])
async def test_model_failure_and_cancellation_preserve_original_semantics(error):
    model = FakeModel([])
    model.complete = AsyncMock(side_effect=error)
    agent = Agent(model, FakeWorkspace())
    if isinstance(error, TimeoutError):
        await agent.run("task")
    else:
        with pytest.raises(type(error)):
            await agent.run("task")
    events = trace(agent)
    assert [event.type for event in events[:3]] == ["RunStarted", "BudgetUpdated", "ModelRequested"]
    assert not any(event.type == "ModelResponded" for event in events)
    assert events[-1].payload.status == agent.state.status
    assert "private" not in events[-1].model_dump_json()


async def test_invalid_tool_is_traced_without_execution():
    agent = Agent(FakeModel([response()]), FakeWorkspace())
    with pytest.raises(AgentProtocolError):
        await agent.run("task")
    assert [event.type for event in trace(agent)[-3:]] == ["ToolCalled", "ToolFailed", "RunFailed"]


async def test_budget_termination_is_finished_with_exact_status():
    agent = Agent(FakeModel([response()]), FakeWorkspace(),
                  policy=ActionPolicy(tool_risks={"record": Risk.LOW}),
                  tool_registry=ToolRegistry([RecordingTool()]), budgets=BudgetLimits(max_steps=1))
    await agent.run("task")
    assert trace(agent)[-1].type == "RunFinished"
    assert trace(agent)[-1].payload.status == "max_steps"
    assert trace(agent)[-1].payload.usage == Usage(input_tokens=4, output_tokens=2, cost=0.2)


@pytest.mark.parametrize("fail_type", ["RunStarted", "ModelRequested", "ToolCalled", "RunFinished"])
async def test_sink_failure_is_not_silent_and_prevents_unrecorded_actions(fail_type):
    store = MemoryEventStore()
    original_append = store.append

    def append(event):
        if event.type == fail_type:
            raise OSError("disk full")
        original_append(event)

    store.append = append
    tool = RecordingTool()
    model = FakeModel([response(), ModelResponse(content="done")])
    model.complete = AsyncMock(wraps=model.complete)
    agent = Agent(model, FakeWorkspace(), policy=ActionPolicy(tool_risks={"record": Risk.LOW}),
                  tool_registry=ToolRegistry([tool]), event_store=store)
    with pytest.raises(EventStoreError):
        await agent.run("task")
    assert agent.state.status is RunStatus.FAILED
    if fail_type in {"RunStarted", "ModelRequested"}:
        model.complete.assert_not_awaited()
    if fail_type != "RunFinished":
        assert tool.calls == []


async def test_failed_terminal_write_does_not_replace_model_exception():
    store = MemoryEventStore()
    original_append = store.append

    def append(event):
        if event.type == "RunFailed":
            raise OSError("disk full")
        original_append(event)

    store.append = append
    agent = Agent(FakeModel([]), FakeWorkspace(), event_store=store)
    with pytest.raises(RuntimeError, match="no responses remaining") as captured:
        await agent.run("task")
    assert captured.value.__notes__ == ["The terminal event could not be recorded."]