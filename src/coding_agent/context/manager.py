"""Pure model-request history boundary, independent of providers and storage."""

from typing import Protocol

from coding_agent.models.base import Message, ToolDefinition


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