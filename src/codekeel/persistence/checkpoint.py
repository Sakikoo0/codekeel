"""Versioned, data-only checkpoints; never deserialize executable runtime objects."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from codekeel.agent.state import AgentState, RunStatus
from codekeel.context.compaction import _turns
from codekeel.events.models import RunID
from codekeel.models.base import ToolDefinition
from codekeel.planning import Plan
from codekeel.runtime.approvals import PendingApproval
from codekeel.runtime.budgets import BudgetLimits
from codekeel.runtime.policy import ActionPolicy


class WorkspaceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["local", "docker", "custom"]
    root: str = Field(min_length=1)


class RunMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str | None = None


class Checkpoint(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: Literal[1] = 1
    run_id: RunID
    revision: int = Field(ge=1, strict=True)
    ready: bool = Field(strict=True)
    event_sequence: int = Field(ge=1, strict=True)
    last_event_id: str = Field(min_length=1)
    state: AgentState
    budgets: BudgetLimits
    elapsed_seconds: float = Field(ge=0)
    workspace: WorkspaceMetadata
    metadata: RunMetadata = Field(default_factory=RunMetadata)
    tools: list[ToolDefinition]
    policy: ActionPolicy = Field(default_factory=ActionPolicy)
    pending_approval: PendingApproval | None = None
    # Opaque plan data only, required by Commit 17; no planner is implemented.
    plan: Plan | None = None
    # Context summaries are already canonical messages in state.messages.

    @model_validator(mode="after")
    def validate_boundary(self):
        if self.ready:
            pending = self.pending_approval
            waiting = self.state.status is RunStatus.WAITING_FOR_APPROVAL
            if waiting != (pending is not None):
                raise ValueError("Waiting state requires exactly one pending approval")
            if pending is not None:
                messages = self.state.messages
                if (not messages or messages[-1].role != "assistant"
                        or messages[-1].tool_call_id is not None
                        or messages[-1].tool_calls != [pending.call]):
                    raise ValueError("Pending action must match the unmatched final assistant call")
                _turns(messages[:-1])
            else:
                _turns(self.state.messages)
        if self.state.steps > self.state.model_calls or self.state.tool_calls > self.state.steps:
            raise ValueError("Inconsistent persisted counters")
        return self