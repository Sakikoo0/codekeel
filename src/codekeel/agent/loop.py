"""Minimal bounded linear coding-agent control loop."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from pydantic import JsonValue

from codekeel.agent.state import AgentState, RunStatus
from codekeel.agent.termination import TerminationPolicy
from codekeel.context.compaction import DeterministicContextManager, estimate_context_tokens
from codekeel.context.manager import ContextManager, SummaryContextManager
from codekeel.context.repo import RepoContext
from codekeel.context.tool_output import ToolOutputManager
from codekeel.events.models import (
    EVENT_ADAPTER,
    BudgetUpdated,
    ContextCompacted,
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
from codekeel.events.store import EventStore, EventStoreError, MemoryEventStore
from codekeel.models.base import Message, Model, ModelResponse, ToolCall, ToolDefinition, ToolResult
from codekeel.persistence.checkpoint import Checkpoint, RunMetadata, WorkspaceMetadata
from codekeel.persistence.store import CheckpointStore, PersistenceError, ResumeError
from codekeel.runtime.budgets import BudgetLimits
from codekeel.tools import ToolArgumentsError, ToolContext, ToolRegistry, UnknownToolError, default_tool_registry
from codekeel.workspace.base import Workspace

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
        checkpoint_store: CheckpointStore | None = None,
        workspace_metadata: WorkspaceMetadata | None = None,
        run_metadata: RunMetadata | None = None,
        plan: JsonValue = None,
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
        if checkpoint_store is not None and workspace_metadata is None:
            raise ValueError("Persistence requires explicit workspace metadata")
        if checkpoint_store is not None and event_store is None:
            raise ValueError("Persistence requires an explicit event store")
        self.checkpoint_store = checkpoint_store
        self.workspace_metadata = workspace_metadata
        self.run_metadata = run_metadata if run_metadata is not None else RunMetadata()
        self.plan = plan
        self._checkpoint_revision = 0
        self._last_event_id = ""
        self._at_boundary = False
        self.run_id: str | None = None
        self._event_sequence = 0
        self._tool_context: ToolContext | None = None
        self._started_at: float | None = None

    async def run(self, task: str, *, repo_context: RepoContext | None = None) -> AgentState:
        if self.checkpoint_store is not None:
            await self._validate_workspace()
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
        self._checkpoint_revision = 0
        self._last_event_id = ""
        self._at_boundary = True
        self._tool_context = ToolContext(workspace=self.workspace, run_id=self.run_id, defer_output_limits=True)
        self._started_at = self.termination_policy.clock()
        return await self._drive(new=True)

    async def resume(self, run_id: str) -> AgentState:
        """Restore a settled checkpoint and continue the next model step, never replay."""
        if self.checkpoint_store is None:
            raise ResumeError("Resume requires a checkpoint store")
        checkpoint = self.checkpoint_store.load(run_id)
        if not checkpoint.ready:
            raise ResumeError("Run was interrupted during a step; automatic replay is unsafe")
        if checkpoint.workspace != self.workspace_metadata:
            raise ResumeError("Workspace metadata does not match the checkpoint")
        await self._validate_workspace()
        if checkpoint.tools != self.tool_registry.definitions():
            raise ResumeError("Tool definitions do not match the checkpoint")
        trace = self.event_store.read(run_id)
        if (trace.warning or not trace.events or len(trace.events) != checkpoint.event_sequence
                or trace.events[-1].event_id != checkpoint.last_event_id):
            raise ResumeError("Checkpoint and event log do not share the same settled frontier")
        start = trace.events[0]
        if not isinstance(start, RunStarted) or start.payload.budgets != checkpoint.budgets:
            raise ResumeError("Persisted budgets do not match the original run")
        accounting = next((event.payload for event in reversed(trace.events)
                           if isinstance(event, (BudgetUpdated, RunFinished, RunFailed))), None)
        expected = {"usage": checkpoint.state.usage, "steps": checkpoint.state.steps,
                    "model_calls": checkpoint.state.model_calls, "tool_calls": checkpoint.state.tool_calls}
        if accounting is not None and any(getattr(accounting, key) != value for key, value in expected.items()):
            raise ResumeError("Checkpoint accounting does not match the trace")
        if accounting is None and (checkpoint.state.model_calls or checkpoint.state.tool_calls
                                   or checkpoint.state.steps or checkpoint.state.usage != AgentState().usage):
            raise ResumeError("Initial checkpoint has unrecorded usage")
        if isinstance(trace.events[-1], RunFinished) and trace.events[-1].payload.status != checkpoint.state.status:
            raise ResumeError("Checkpoint terminal status does not match the trace")
        self.state = checkpoint.state.model_copy(deep=True)
        if self.state.status not in {
            RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.MAX_STEPS, RunStatus.MAX_COST, RunStatus.MAX_TOKENS,
        }:
            raise ResumeError("Checkpoint status cannot be resumed")
        self.run_id = checkpoint.run_id
        self._event_sequence = checkpoint.event_sequence
        self._last_event_id = checkpoint.last_event_id
        self._checkpoint_revision = checkpoint.revision
        self.run_metadata = checkpoint.metadata
        self.plan = checkpoint.plan
        self.termination_policy = TerminationPolicy(budgets=checkpoint.budgets, clock=self.termination_policy.clock)
        self._started_at = self.termination_policy.clock() - checkpoint.elapsed_seconds
        self._tool_context = ToolContext(workspace=self.workspace, run_id=run_id, defer_output_limits=True)
        self._at_boundary = True
        if self.state.status is not RunStatus.RUNNING:
            # A process may die after saving its terminal step but before writing
            # RunFinished. Complete that bookkeeping once, without calling a model.
            if not isinstance(trace.events[-1], RunFinished):
                self._at_boundary = False
                self.save_checkpoint()
                self._emit(RunFinished, status=self.state.status, **self._budget_payload())
                self._at_boundary = True
                self.save_checkpoint()
            return self.state
        return await self._drive(new=False)

    async def _validate_workspace(self) -> None:
        assert self.workspace_metadata is not None
        info = await self.workspace.inspect_path(".")
        if not info.exists or not info.is_directory or info.canonical_path != ".":
            raise ResumeError("Workspace root is missing or is not a canonical directory")

    def save_checkpoint(self) -> None:
        """Save the current boundary (including opaque plan and summary messages)."""
        if self.checkpoint_store is None:
            return
        assert self.workspace_metadata is not None and self.run_id is not None and self._started_at is not None
        try:
            checkpoint = Checkpoint(
                run_id=self.run_id, revision=self._checkpoint_revision + 1, ready=self._at_boundary,
                event_sequence=self._event_sequence, last_event_id=self._last_event_id,
                state=self.state, budgets=self.termination_policy.budgets,
                elapsed_seconds=max(0.0, self.termination_policy.clock() - self._started_at),
                workspace=self.workspace_metadata, metadata=self.run_metadata,
                tools=self.tool_registry.definitions(), plan=self.plan,
            )
            self.checkpoint_store.save(checkpoint, expected_revision=self._checkpoint_revision)
        except Exception as error:
            raise PersistenceError("Unable to save checkpoint") from error
        self._checkpoint_revision = checkpoint.revision

    async def _drive(self, *, new: bool) -> AgentState:
        try:
            if new:
                self._emit(RunStarted, messages=self.state.messages, budgets=self.termination_policy.budgets)
                self.save_checkpoint()
            while self.state.status is RunStatus.RUNNING:
                await self.step()
            self._emit(RunFinished, status=self.state.status, **self._budget_payload())
            self.save_checkpoint()
        except (EventStoreError, PersistenceError):
            # A broken sink cannot reliably record its own failure; stop immediately.
            self.state.status = RunStatus.FAILED
            raise
        except asyncio.CancelledError as error:
            self.state.status = RunStatus.CANCELLED
            self._record_terminal_error(error, cancelled=True)
            self._save_interrupted(error)
            raise
        except Exception as error:
            self.state.status = RunStatus.FAILED
            self._record_terminal_error(error)
            self._save_interrupted(error)
            raise
        return self.state

    def _save_interrupted(self, error: BaseException) -> None:
        self._at_boundary = False
        try:
            self.save_checkpoint()
        except PersistenceError:
            error.add_note("The interrupted checkpoint could not be recorded.")

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
        self._last_event_id = event.event_id

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
        # Persist an in-flight marker before any model/tool side effects. CAS
        # stops a stale second runtime before it can repeat the same step.
        self._at_boundary = False
        self.save_checkpoint()
        await self._step()
        self._at_boundary = self.state.status in {
            RunStatus.RUNNING, RunStatus.COMPLETED, RunStatus.MAX_STEPS, RunStatus.MAX_COST, RunStatus.MAX_TOKENS,
        }
        self.save_checkpoint()

    async def _step(self) -> None:
        if self._tool_context is None or self._started_at is None:
            raise RuntimeError("Agent step requires an active run")
        if status := self.termination_policy.evaluate(self.state, started_at=self._started_at):
            self.state.status = status
            return

        definitions = self.tool_registry.definitions()
        try:
            await self._prepare_context(definitions)
        except TimeoutError:
            self.state.status = RunStatus.TIMEOUT
            return
        # Summary requests share all parent limits; approval before compaction is
        # not permission to send another request after compaction spends usage.
        if status := self.termination_policy.evaluate(self.state, started_at=self._started_at):
            self.state.status = status
            return
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

    async def _prepare_context(self, definitions: list[ToolDefinition]) -> None:
        manager = self.context_manager
        limit = self.termination_policy.budgets.max_model_calls
        # Reserve the pending main request. With only one slot left, use the
        # deterministic path rather than silently exceeding the parent's ceiling.
        can_summarize = limit is None or self.state.model_calls + 1 < limit
        if isinstance(manager, SummaryContextManager) and can_summarize:
            request = manager.plan_summary(self.state.messages, tools=definitions)
            if request is not None:
                assert self._started_at is not None
                if status := self.termination_policy.evaluate(self.state, started_at=self._started_at):
                    self.state.status = status
                    return
                self.state.model_calls += 1
                self._emit(BudgetUpdated, **self._budget_payload())
                self._emit(ModelRequested, model_call=self.state.model_calls, messages=request.messages, tools=[])
                response = await self._complete_with_deadline(
                    [], model=request.model, messages=request.messages,
                )
                # Charge even a malformed, tool-calling or oversized summary.
                self.state.add_usage(response.usage)
                self._emit(ModelResponded, model_call=self.state.model_calls, response=response)
                self._emit(BudgetUpdated, **self._budget_payload())
                prepared = manager.apply_summary(request, response)
                self._emit(
                    ContextCompacted, before_estimated_tokens=request.before_estimated_tokens,
                    after_estimated_tokens=estimate_context_tokens(prepared, definitions),
                    messages_removed=request.messages_removed, summary_model_calls=1,
                )
                self.state.messages = prepared
                return
        self.state.messages = manager.prepare(self.state.messages, tools=definitions)

    async def _complete_with_deadline(
        self, definitions: list[ToolDefinition], *, model: Model | None = None, messages: list[Message] | None = None,
    ) -> ModelResponse:
        assert self._started_at is not None
        selected_model = model if model is not None else self.model
        selected_messages = messages if messages is not None else self.state.messages
        remaining = self.termination_policy.remaining_wall_time(started_at=self._started_at)
        if remaining is None:
            return await selected_model.complete(selected_messages, tools=definitions)
        if remaining <= 0:
            raise TimeoutError
        async with asyncio.timeout(remaining):
            return await selected_model.complete(selected_messages, tools=definitions)

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