"""Pure model-request history boundary, independent of providers and storage."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from codekeel.models.base import Message, Model, ModelResponse, ToolDefinition


class ContextBudgetExceeded(ValueError):
    """Protected history and tool definitions cannot fit the configured budget."""


class ContextHistoryError(ValueError):
    """History contains an incomplete or ambiguous tool exchange."""


class ContextManager(Protocol):
    def prepare(
        self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
    ) -> list[Message]:
        """Return independent, bounded history for the next model request."""
        ...


@dataclass(frozen=True)
class SummaryRequest:
    """A pure compaction plan; the parent runtime owns the actual model call."""

    messages: list[Message]
    retained: list[Message]
    insertion_index: int
    before_estimated_tokens: int
    messages_removed: int
    tools: list[ToolDefinition]
    model: Model | None = None


@runtime_checkable
class SummaryContextManager(ContextManager, Protocol):
    def plan_summary(
        self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
    ) -> SummaryRequest | None: ...

    def apply_summary(self, request: SummaryRequest, response: ModelResponse) -> list[Message]: ...