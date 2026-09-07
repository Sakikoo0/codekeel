"""Contracts shared by tools and the tool runtime."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from codekeel.models.base import ToolDefinition, ToolResult
from codekeel.planning import Plan
from codekeel.workspace.base import Workspace


class ToolRegistryError(ValueError):
    """Base class for invalid registration or dispatch requests."""


class DuplicateToolError(ToolRegistryError):
    """Raised when registration would replace an existing tool."""


class UnknownToolError(ToolRegistryError):
    """Raised when a call names a tool that is not registered."""


class ToolArgumentsError(ToolRegistryError):
    """Raised when tool arguments do not match the advertised schema."""


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Per-run dependencies available to every tool invocation."""

    workspace: Workspace
    run_id: str
    # Set by Agent only when a ToolOutputManager will bound the returned content.
    defer_output_limits: bool = False
    update_plan: Callable[[Plan], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("run_id cannot be blank")


class Tool(Protocol):
    """Executable tool with an explicit model-facing schema."""

    name: str
    description: str

    def definition(self) -> ToolDefinition:
        """Return the schema advertised to the model."""
        ...

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute one validated invocation in the supplied run context."""
        ...