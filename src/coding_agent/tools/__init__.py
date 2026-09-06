"""Typed tools and registry dispatch."""

from coding_agent.tools.base import Tool, ToolContext
from coding_agent.tools.filesystem import (
    EditFileTool,
    FileSystemConfig,
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
    filesystem_tools,
)
from coding_agent.tools.registry import (
    DuplicateToolError,
    ToolArgumentsError,
    ToolRegistry,
    ToolRegistryError,
    UnknownToolError,
    default_tool_registry,
)
from coding_agent.tools.shell import (
    DEFAULT_DENIED_COMMANDS,
    DEFAULT_ENV_DENY_PATTERNS,
    ShellConfig,
    ShellTool,
)

__all__ = [
    "DuplicateToolError",
    "DEFAULT_DENIED_COMMANDS",
    "DEFAULT_ENV_DENY_PATTERNS",
    "EditFileTool",
    "FileSystemConfig",
    "FindFilesTool",
    "ListDirectoryTool",
    "ReadFileTool",
    "SearchFilesTool",
    "ShellTool",
    "ShellConfig",
    "Tool",
    "ToolArgumentsError",
    "ToolContext",
    "ToolRegistry",
    "ToolRegistryError",
    "UnknownToolError",
    "WriteFileTool",
    "default_tool_registry",
    "filesystem_tools",
]