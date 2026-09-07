"""Typed state for an agent run."""

from enum import StrEnum

from pydantic import BaseModel, Field

from codekeel.models.base import Message, Usage


class RunStatus(StrEnum):
    """Lifecycle and terminal states for an agent run."""

    IDLE = "idle"
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    VERIFICATION_FAILED = "verification_failed"
    MAX_STEPS = "max_steps"
    MAX_COST = "max_cost"
    MAX_TOKENS = "max_tokens"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class AgentState(BaseModel):
    """Conversation history, lifecycle status, and accumulated model usage."""

    messages: list[Message] = Field(default_factory=list)
    status: RunStatus = RunStatus.IDLE
    usage: Usage = Field(default_factory=Usage)
    steps: int = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)

    verification_attempts: int = Field(default=0, ge=0, strict=True)
    verification_commands: int = Field(default=0, ge=0, strict=True)
    verification_passed: bool | None = Field(default=None, strict=True)

    def add_usage(self, usage: Usage) -> None:
        self.usage = self.usage + usage