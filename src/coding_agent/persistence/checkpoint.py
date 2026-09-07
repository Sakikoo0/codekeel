"""Versioned, data-only checkpoints; never deserialize executable runtime objects."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, model_validator

from coding_agent.agent.state import AgentState
from coding_agent.context.compaction import _turns
from coding_agent.events.models import RunID
from coding_agent.models.base import ToolDefinition
from coding_agent.runtime.budgets import BudgetLimits


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
    # Opaque plan data only, required by Commit 17; no planner is implemented.
    plan: JsonValue = None
    # Context summaries are already canonical messages in state.messages.

    @model_validator(mode="after")
    def validate_boundary(self):
        if self.ready:
            _turns(self.state.messages)
        if self.state.steps > self.state.model_calls or self.state.tool_calls > self.state.steps:
            raise ValueError("Inconsistent persisted counters")
        return self