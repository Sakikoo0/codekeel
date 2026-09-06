"""Minimal bounded linear coding-agent control loop."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

from coding_agent.agent.state import AgentState, RunStatus
from coding_agent.agent.termination import TerminationPolicy
from coding_agent.models.base import Message, Model, ModelResponse, ToolCall, ToolResult
from coding_agent.runtime.budgets import BudgetLimits
from coding_agent.tools import ToolArgumentsError, ToolContext, ToolRegistry, UnknownToolError, default_tool_registry
from coding_agent.workspace.base import Workspace

_DEFAULT_SYSTEM_PROMPT = "You are a coding agent. Use the available tools when needed, then return a final answer."


class AgentProtocolError(ValueError):
    """Raised when a model response cannot be handled by the baseline loop."""


@dataclass(frozen=True, slots=True)
class FinalAnswer:
    """A model response that completes the run."""

    content: str


class Agent:
    """Run a model until it returns a final answer."""

    def __init__(
        self,
        model: Model,
        workspace: Workspace,
        *,
        tool_registry: ToolRegistry | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        budgets: BudgetLimits | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model = model
        self.workspace = workspace
        self.tool_registry = tool_registry or default_tool_registry()
        self.system_prompt = system_prompt
        self.termination_policy = TerminationPolicy(
            budgets=budgets if budgets is not None else BudgetLimits(),
            clock=clock,
        )
        self.state = AgentState()
        self._tool_context: ToolContext | None = None
        self._started_at: float | None = None

    async def run(self, task: str) -> AgentState:
        self.state = AgentState(
            messages=[
                Message(role="system", content=self.system_prompt),
                Message(role="user", content=task),
            ],
            status=RunStatus.RUNNING,
        )
        self._tool_context = ToolContext(workspace=self.workspace, run_id=uuid4().hex)
        self._started_at = self.termination_policy.clock()
        try:
            while self.state.status is RunStatus.RUNNING:
                await self.step()
        except asyncio.CancelledError:
            self.state.status = RunStatus.CANCELLED
            raise
        except Exception:
            self.state.status = RunStatus.FAILED
            raise
        return self.state

    async def step(self) -> None:
        if self._tool_context is None or self._started_at is None:
            raise RuntimeError("Agent step requires an active run")
        if status := self.termination_policy.evaluate(self.state, started_at=self._started_at):
            self.state.status = status
            return

        self.state.steps += 1
        self.state.model_calls += 1
        try:
            response = await self._complete_with_deadline()
        except TimeoutError:
            self.state.status = RunStatus.TIMEOUT
            return
        self.state.add_usage(response.usage)
        self.state.messages.append(
            Message(role="assistant", content=response.content, tool_calls=response.tool_calls)
        )

        action = _parse_action(response)
        if isinstance(action, FinalAnswer):
            self.state.status = RunStatus.COMPLETED
            return

        if self.termination_policy.wall_time_exceeded(started_at=self._started_at):
            self.state.status = RunStatus.TIMEOUT
            return
        self.state.tool_calls += 1
        try:
            result = await self._execute_tool_with_deadline(action)
        except TimeoutError:
            self.state.status = RunStatus.TIMEOUT
            return
        except (UnknownToolError, ToolArgumentsError) as error:
            raise AgentProtocolError(str(error)) from error
        self.state.messages.append(
            Message(
                role="tool",
                content=result.model_dump_json(),
                tool_call_id=action.id,
            )
        )

    async def _complete_with_deadline(self) -> ModelResponse:
        assert self._started_at is not None
        remaining = self.termination_policy.remaining_wall_time(started_at=self._started_at)
        if remaining is None:
            return await self.model.complete(self.state.messages, tools=self.tool_registry.definitions())
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            return await self.model.complete(self.state.messages, tools=self.tool_registry.definitions())

    async def _execute_tool_with_deadline(self, action: ToolCall) -> ToolResult:
        assert self._started_at is not None
        assert self._tool_context is not None
        remaining = self.termination_policy.remaining_wall_time(started_at=self._started_at)
        if remaining is None:
            return await self.tool_registry.execute(action, self._tool_context)
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            return await self.tool_registry.execute(action, self._tool_context)


def _parse_action(response: ModelResponse) -> ToolCall | FinalAnswer:
    if not response.tool_calls:
        if response.content is None:
            raise AgentProtocolError("Final response must include content")
        return FinalAnswer(content=response.content)

    if len(response.tool_calls) != 1:
        raise AgentProtocolError("Baseline agent requires exactly one tool call per response")

    return response.tool_calls[0]