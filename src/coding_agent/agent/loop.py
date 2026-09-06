"""Minimal bounded linear coding-agent control loop."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from coding_agent.agent.state import AgentState, RunStatus
from coding_agent.agent.termination import TerminationPolicy
from coding_agent.context.compaction import DeterministicContextManager
from coding_agent.context.manager import ContextManager
from coding_agent.context.repo import RepoContext
from coding_agent.context.tool_output import ToolOutputManager
from coding_agent.events.models import (
    EVENT_ADAPTER,
    BudgetUpdated,
    EventEnvelope,
    ModelRequested,
    ModelResponded,
    RunFailed,
    RunFinished,
    RunStarted,
    ToolCalled,
    ToolCompleted,
    ToolFailed,
)
from coding_agent.events.store import EventStore, EventStoreError, MemoryEventStore
from coding_agent.models.base import Message, Model, ModelResponse, ToolCall, ToolDefinition, ToolResult
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
        event_store: EventStore | None = None,
        tool_output_manager: ToolOutputManager | None = None,
        context_manager: ContextManager | None = None,
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
        self.event_store = event_store if event_store is not None else MemoryEventStore()
        self.tool_output_manager = tool_output_manager if tool_output_manager is not None else ToolOutputManager()
        self.context_manager = context_manager if context_manager is not None else DeterministicContextManager()
        self.run_id: str | None = None
        self._event_sequence = 0
        self._tool_context: ToolContext | None = None
        self._started_at: float | None = None

    async def run(self, task: str, *, repo_context: RepoContext | None = None) -> AgentState:
        system_prompt = self.system_prompt
        if repo_context is not None:
            system_prompt += "\n\n" + repo_context.render()
        self.state = AgentState(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=task),
            ],
            status=RunStatus.RUNNING,
        )
        self.run_id = uuid4().hex
        self._event_sequence = 0
        self._tool_context = ToolContext(workspace=self.workspace, run_id=self.run_id, defer_output_limits=True)
        self._started_at = self.termination_policy.clock()
        try:
            self._emit(RunStarted, messages=self.state.messages, budgets=self.termination_policy.budgets)
            while self.state.status is RunStatus.RUNNING:
                await self.step()
            self._emit(RunFinished, status=self.state.status, **self._budget_payload())
        except EventStoreError:
            # A broken sink cannot reliably record its own failure; stop immediately.
            self.state.status = RunStatus.FAILED
            raise
        except asyncio.CancelledError as error:
            self.state.status = RunStatus.CANCELLED
            self._record_terminal_error(error, cancelled=True)
            raise
        except Exception as error:
            self.state.status = RunStatus.FAILED
            self._record_terminal_error(error)
            raise
        return self.state

    def _budget_payload(self) -> dict[str, Any]:
        return {
            "usage": self.state.usage, "steps": self.state.steps,
            "model_calls": self.state.model_calls, "tool_calls": self.state.tool_calls,
        }

    def _emit(self, event_class: type[EventEnvelope], **payload: Any) -> None:
        assert self.run_id is not None
        try:
            event = EVENT_ADAPTER.validate_python({
                "type": event_class.model_fields["type"].default,
                "run_id": self.run_id, "sequence": self._event_sequence + 1, "payload": payload,
            })
            self.event_store.append(event)
        except Exception as error:
            raise EventStoreError("Unable to append runtime event") from error
        self._event_sequence += 1

    def _record_terminal_error(self, error: BaseException, *, cancelled: bool = False) -> None:
        try:
            if cancelled:
                self._emit(RunFinished, status=self.state.status, **self._budget_payload())
            else:
                self._emit(
                    RunFailed, status=self.state.status, error_type=type(error).__name__, **self._budget_payload(),
                )
        except EventStoreError:
            error.add_note("The terminal event could not be recorded.")

    async def step(self) -> None:
        if self._tool_context is None or self._started_at is None:
            raise RuntimeError("Agent step requires an active run")
        if status := self.termination_policy.evaluate(self.state, started_at=self._started_at):
            self.state.status = status
            return

        definitions = self.tool_registry.definitions()
        self.state.messages = self.context_manager.prepare(self.state.messages, tools=definitions)
        self.state.steps += 1
        self.state.model_calls += 1
        self._emit(BudgetUpdated, **self._budget_payload())
        self._emit(
            ModelRequested, model_call=self.state.model_calls,
            messages=self.state.messages, tools=definitions,
        )
        try:
            response = await self._complete_with_deadline(definitions)
        except TimeoutError:
            self.state.status = RunStatus.TIMEOUT
            return
        self.state.add_usage(response.usage)
        self._emit(ModelResponded, model_call=self.state.model_calls, response=response)
        self._emit(BudgetUpdated, **self._budget_payload())
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
        self._emit(BudgetUpdated, **self._budget_payload())
        self._emit(ToolCalled, call=action)
        try:
            result = await self._execute_tool_with_deadline(action)
        except TimeoutError:
            self._emit(ToolFailed, tool_call_id=action.id, error_type="TimeoutError")
            self.state.status = RunStatus.TIMEOUT
            return
        except (UnknownToolError, ToolArgumentsError) as error:
            self._emit(ToolFailed, tool_call_id=action.id, error_type=type(error).__name__)
            raise AgentProtocolError(str(error)) from error
        except (Exception, asyncio.CancelledError) as error:
            try:
                self._emit(ToolFailed, tool_call_id=action.id, error_type=type(error).__name__)
            except EventStoreError:
                error.add_note("The tool failure event could not be recorded.")
            raise
        if result.is_error:
            self._emit(ToolFailed, tool_call_id=action.id, error_type="ToolResultError", result=result)
        else:
            self._emit(ToolCompleted, tool_call_id=action.id, result=result)
        # Trace retains the original result; only the reduced result enters history.
        assert self.run_id is not None
        remaining = self.termination_policy.remaining_wall_time(started_at=self._started_at)
        if remaining is not None and remaining <= 0:
            self.state.status = RunStatus.TIMEOUT
            return
        try:
            async with asyncio.timeout(remaining):
                result = await self.tool_output_manager.process(result, workspace=self.workspace, run_id=self.run_id)
        except TimeoutError:
            self.state.status = RunStatus.TIMEOUT
            return
        self.state.messages.append(
            Message(
                role="tool",
                content=result.model_dump_json(),
                tool_call_id=action.id,
            )
        )

    async def _complete_with_deadline(self, definitions: list[ToolDefinition]) -> ModelResponse:
        assert self._started_at is not None
        remaining = self.termination_policy.remaining_wall_time(started_at=self._started_at)
        if remaining is None:
            return await self.model.complete(self.state.messages, tools=definitions)
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            return await self.model.complete(self.state.messages, tools=definitions)

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