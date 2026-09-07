"""Typed tool implementations and deterministic registry dispatch."""

from collections.abc import Iterable

from codekeel.models.base import ToolCall, ToolDefinition, ToolResult
from codekeel.tools.base import (
    DuplicateToolError,
    Tool,
    ToolContext,
    ToolRegistryError,
    UnknownToolError,
)
from codekeel.tools.base import ToolArgumentsError as ToolArgumentsError
from codekeel.tools.filesystem import filesystem_tools
from codekeel.tools.planning import UpdatePlanTool
from codekeel.tools.shell import ShellTool


class ToolRegistry:
    """Hold tools by name and dispatch typed model calls to them."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        """Register one tool, rejecting ambiguous or inconsistent names."""
        definition = tool.definition()
        if definition.name != tool.name:
            raise ToolRegistryError(
                f"Tool name {tool.name!r} does not match definition name {definition.name!r}"
            )
        if tool.name in self._tools:
            raise DuplicateToolError(f"Tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def definitions(self) -> list[ToolDefinition]:
        """Return model-facing schemas in deterministic registration order."""
        return [tool.definition() for tool in self._tools.values()]

    def get(self, name: str) -> Tool:
        """Return a registered tool or fail without executing anything."""
        try:
            return self._tools[name]
        except KeyError as error:
            raise UnknownToolError(f"Unsupported tool: {name}") from error

    async def execute(self, call: ToolCall, context: ToolContext) -> ToolResult:
        """Dispatch one typed call to the matching tool."""
        tool = self.get(call.name)
        return await tool.execute(dict(call.arguments), context)

def default_tool_registry() -> ToolRegistry:
    """Build the current default tool set in stable model-facing order."""
    return ToolRegistry([ShellTool(), *filesystem_tools(), UpdatePlanTool()])