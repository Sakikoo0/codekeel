"""Pending action data and durable, non-executing approval decisions."""

from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from codekeel.models.base import ToolCall
from codekeel.runtime.policy import Risk

if TYPE_CHECKING:
    from codekeel.events.store import EventStore
    from codekeel.persistence.checkpoint import Checkpoint
    from codekeel.persistence.store import CheckpointStore


class PendingApproval(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    action_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[a-f0-9]{32}$")
    call: ToolCall
    risk: Risk
    approved: bool | None = Field(default=None, strict=True)


def validate_approval(checkpoint: "Checkpoint", event_store: "EventStore") -> None:
    """Bind the pending action and decision to the settled append-only trace."""
    from codekeel.events.models import ApprovalRequested, ApprovalResolved, ModelResponded, RunStarted
    from codekeel.persistence.store import ResumeError

    pending = checkpoint.pending_approval
    trace = event_store.read(checkpoint.run_id)
    if (not checkpoint.ready or trace.warning or not trace.events
            or len(trace.events) != checkpoint.event_sequence
            or trace.events[-1].event_id != checkpoint.last_event_id):
        raise ResumeError("Approval requires a settled checkpoint and matching event frontier")
    if pending is None or checkpoint.state.status != "waiting_for_approval":
        raise ResumeError("Run has no pending approval")
    start = trace.events[0]
    request = next((e for e in reversed(trace.events) if isinstance(e, ApprovalRequested)), None)
    response = next((e for e in reversed(trace.events) if isinstance(e, ModelResponded)), None)
    if (not isinstance(start, RunStarted) or start.payload.policy != checkpoint.policy
            or request is None or request.payload != pending.model_copy(update={"approved": None})
            or response is None or response.payload.response.tool_calls != [pending.call]):
        raise ResumeError("Pending approval does not match the recorded action or policy")
    last = trace.events[-1]
    if pending.approved is None:
        if last != request:
            raise ResumeError("Unresolved approval has an unexpected event frontier")
    elif not isinstance(last, ApprovalResolved) or last.payload != pending:
        raise ResumeError("Approval decision does not match the event log")


def resolve_approval(
    store: "CheckpointStore", event_store: "EventStore", run_id: str, action_id: str, *, approved: bool,
) -> PendingApproval:
    """Record one decision with CAS ownership; execution requires a separate resume.

    Claim before appending an event. A crash between stores leaves an unready run,
    which cannot execute automatically. Concurrent or repeated decisions fail closed.
    """
    from codekeel.events.models import ApprovalResolved
    from codekeel.persistence.checkpoint import Checkpoint
    from codekeel.persistence.store import ResumeError

    if type(approved) is not bool:
        raise ValueError("Approval decision must be a bool")
    checkpoint = store.load(run_id)
    validate_approval(checkpoint, event_store)
    pending = checkpoint.pending_approval
    assert pending is not None
    if pending.action_id != action_id or pending.approved is not None:
        raise ResumeError("Action ID does not match an unresolved approval")
    claimed = checkpoint.model_copy(update={"revision": checkpoint.revision + 1, "ready": False})
    store.save(claimed, expected_revision=checkpoint.revision)
    resolved = pending.model_copy(update={"approved": approved})
    event = ApprovalResolved(run_id=run_id, sequence=checkpoint.event_sequence + 1, payload=resolved)
    event_store.append(event)
    data = claimed.model_dump()
    data.update(revision=claimed.revision + 1, ready=True, pending_approval=resolved,
                event_sequence=event.sequence, last_event_id=event.event_id)
    store.save(Checkpoint.model_validate(data), expected_revision=claimed.revision)
    return resolved