"""Typed runtime events and forward-compatible trace decoding."""

from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

from codekeel.models.base import Message, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from codekeel.planning import Plan
from codekeel.runtime.approvals import PendingApproval
from codekeel.runtime.budgets import BudgetLimits
from codekeel.runtime.policy import ActionPolicy

RunID = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]


class Payload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StartPayload(Payload):
    messages: list[Message]
    budgets: BudgetLimits
    policy: ActionPolicy = Field(default_factory=ActionPolicy)
    plan: Plan | None = None


class RequestPayload(Payload):
    model_call: int = Field(ge=1)
    messages: list[Message]
    tools: list[ToolDefinition]


class ResponsePayload(Payload):
    model_call: int = Field(ge=1)
    response: ModelResponse


class CalledPayload(Payload):
    call: ToolCall


class CompletedPayload(Payload):
    tool_call_id: str
    result: ToolResult


class ToolFailurePayload(Payload):
    tool_call_id: str
    error_type: str
    result: ToolResult | None = None


class BudgetPayload(Payload):
    usage: Usage
    steps: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)


class FinishPayload(BudgetPayload):
    status: str = Field(min_length=1)


class FailurePayload(FinishPayload):
    error_type: str


class CompactionPayload(Payload):
    before_estimated_tokens: int = Field(ge=0, strict=True)
    after_estimated_tokens: int = Field(ge=0, strict=True)
    messages_removed: int = Field(ge=1, strict=True)
    summary_model_calls: int = Field(ge=1, strict=True)


class EventEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    run_id: RunID
    sequence: int = Field(ge=1, strict=True)
    timestamp: AwareDatetime = Field(default_factory=lambda: datetime.now(UTC))


class RunStarted(EventEnvelope):
    type: Literal["RunStarted"] = "RunStarted"
    payload: StartPayload


class ModelRequested(EventEnvelope):
    type: Literal["ModelRequested"] = "ModelRequested"
    payload: RequestPayload


class ModelResponded(EventEnvelope):
    type: Literal["ModelResponded"] = "ModelResponded"
    payload: ResponsePayload


class ToolCalled(EventEnvelope):
    type: Literal["ToolCalled"] = "ToolCalled"
    payload: CalledPayload


class ToolCompleted(EventEnvelope):
    type: Literal["ToolCompleted"] = "ToolCompleted"
    payload: CompletedPayload


class ToolFailed(EventEnvelope):
    type: Literal["ToolFailed"] = "ToolFailed"
    payload: ToolFailurePayload


class BudgetUpdated(EventEnvelope):
    type: Literal["BudgetUpdated"] = "BudgetUpdated"
    payload: BudgetPayload


class RunFinished(EventEnvelope):
    type: Literal["RunFinished"] = "RunFinished"
    payload: FinishPayload


class RunFailed(EventEnvelope):
    type: Literal["RunFailed"] = "RunFailed"
    payload: FailurePayload


class ContextCompacted(EventEnvelope):
    type: Literal["ContextCompacted"] = "ContextCompacted"
    payload: CompactionPayload


class PlanPayload(Payload):
    plan: Plan


class PlanUpdated(EventEnvelope):
    type: Literal["PlanUpdated"] = "PlanUpdated"
    payload: PlanPayload


class ApprovalRequested(EventEnvelope):
    type: Literal["ApprovalRequested"] = "ApprovalRequested"
    payload: PendingApproval


class ApprovalResolved(EventEnvelope):
    type: Literal["ApprovalResolved"] = "ApprovalResolved"
    payload: PendingApproval


Event = Annotated[
    RunStarted | ModelRequested | ModelResponded | ToolCalled | ToolCompleted
    | ToolFailed | BudgetUpdated | RunFinished | RunFailed | ContextCompacted | ApprovalRequested | ApprovalResolved
    | PlanUpdated,
    Field(discriminator="type"),
]
EVENT_ADAPTER = TypeAdapter(Event)
_KNOWN_TYPES = frozenset(EVENT_ADAPTER.json_schema()["discriminator"]["mapping"])


class UnknownEvent(EventEnvelope):
    """Preserve unknown type, payload and additional envelope fields for inspection."""

    model_config = ConfigDict(frozen=True, extra="allow")
    type: str = Field(min_length=1)
    payload: JsonValue


def parse_event(data: object) -> Event | UnknownEvent:
    if isinstance(data, dict) and isinstance(data.get("type"), str) and data["type"] not in _KNOWN_TYPES:
        return UnknownEvent.model_validate(data)
    return EVENT_ADAPTER.validate_python(data)