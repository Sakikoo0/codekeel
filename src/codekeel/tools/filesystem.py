"""Workspace-rooted filesystem tools with deterministic safety policies."""

import fnmatch
import re
from dataclasses import dataclass
from pathlib import PurePath, PurePosixPath, PureWindowsPath
from typing import Any

from codekeel.models.base import ToolDefinition, ToolResult
from codekeel.tools.base import Tool, ToolArgumentsError, ToolContext
from codekeel.workspace.models import FileInfo

_DEFAULT_PROTECTED_PATTERNS = (
    ".git",
    ".git/*",
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "**/secrets*",
)
_RECOVERABLE_ERRORS = (
    FileNotFoundError,
    IsADirectoryError,
    NotADirectoryError,
    PermissionError,
    UnicodeError,
    ValueError,
)

@dataclass(frozen=True, slots=True)
class FileSystemConfig:
    """Shared access rules and result limits for filesystem tools."""

    allowed_patterns: tuple[str, ...] = ()
    denied_patterns: tuple[str, ...] = ()
    protected_patterns: tuple[str, ...] = _DEFAULT_PROTECTED_PATTERNS
    max_file_bytes: int = 1_000_000
    max_read_lines: int = 2_000
    max_list_results: int = 1_000
    max_find_results: int = 1_000
    max_search_results: int = 1_000
    read_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "max_file_bytes",
            "max_read_lines",
            "max_list_results",
            "max_find_results",
            "max_search_results",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("allowed_patterns", "denied_patterns", "protected_patterns"):
            patterns = tuple(getattr(self, name))
            if any(not isinstance(pattern, str) or not pattern for pattern in patterns):
                raise ValueError(f"{name} must contain non-empty strings")
            object.__setattr__(self, name, patterns)

    def authorize(self, path: str, *, write: bool = False, check_allowed: bool = True) -> None:
        """Apply policies to an already canonical workspace-relative path."""
        if write and self.read_only:
            raise PermissionError("Filesystem tools are read-only")
        if write and (pattern := _first_match(path, self.protected_patterns)) is not None:
            raise PermissionError(f"Path {path!r} is protected by pattern {pattern!r}")
        if (pattern := _first_match(path, self.denied_patterns)) is not None:
            raise PermissionError(f"Path {path!r} is denied by pattern {pattern!r}")
        if check_allowed and self.allowed_patterns and _first_match(path, self.allowed_patterns) is None:
            raise PermissionError(f"Path {path!r} does not match an allowed pattern")

    def is_accessible(self, path: str) -> bool:
        try:
            self.authorize(path)
        except PermissionError:
            return False
        return True

@dataclass(frozen=True, slots=True)
class ReadFileTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "read_file"
    description: str = "Read a bounded UTF-8 text file from the workspace."

    def definition(self) -> ToolDefinition:
        return _definition(
            self,
            {
                "path": {"type": "string"},
                "offset": {"type": "integer", "minimum": 0},
                "limit": {"type": "integer", "minimum": 1},
            },
            required=["path"],
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(self.name, arguments, required={"path"}, optional={"offset", "limit"})
        path = _non_empty_string(self.name, "path", arguments["path"])
        offset = _non_negative_int(self.name, "offset", arguments.get("offset", 0))
        limit = _positive_int(self.name, "limit", arguments.get("limit", self.config.max_read_lines))
        limit = min(limit, self.config.max_read_lines)
        try:
            info = await context.workspace.inspect_path(path)
            self.config.authorize(info.canonical_path)
            _require_file(info, path)
            if info.size > self.config.max_file_bytes:
                raise ValueError(
                    f"File {path!r} is too large ({info.size} bytes; limit {self.config.max_file_bytes})"
                )
            result = await context.workspace.read_file(path)
            if result.is_binary:
                return ToolResult(content=f"[Binary file: {info.size} bytes; content not returned]")
            lines = result.content.splitlines(keepends=True)
            header = f"[{path} | {len(lines)} lines]\n"
            return ToolResult(content=header + _format_lines(lines, offset, limit))
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)

@dataclass(frozen=True, slots=True)
class WriteFileTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "write_file"
    description: str = "Create or overwrite a bounded UTF-8 text file in the workspace."

    def definition(self) -> ToolDefinition:
        return _definition(
            self,
            {"path": {"type": "string"}, "content": {"type": "string"}},
            required=["path", "content"],
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(self.name, arguments, required={"path", "content"})
        path = _non_empty_string(self.name, "path", arguments["path"])
        content = _string(self.name, "content", arguments["content"])
        try:
            _check_content_size(path, content, self.config.max_file_bytes)
            info = await context.workspace.inspect_path(path)
            self.config.authorize(info.canonical_path, write=True)
            if info.exists and info.is_directory:
                raise IsADirectoryError(f"Workspace path is a directory: {path!r}")
            result = await context.workspace.write_file(path, content)
            return ToolResult(
                content=f"Wrote {len(content)} characters ({len(content.splitlines())} lines) to {result.path}."
            )
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)


@dataclass(frozen=True, slots=True)
class EditFileTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "edit_file"
    description: str = "Replace one unique text occurrence in a workspace file."

    def definition(self) -> ToolDefinition:
        return _definition(
            self,
            {
                "path": {"type": "string"},
                "old_text": {"type": "string", "minLength": 1},
                "new_text": {"type": "string"},
            },
            required=["path", "old_text", "new_text"],
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(self.name, arguments, required={"path", "old_text", "new_text"})
        path = _non_empty_string(self.name, "path", arguments["path"])
        old_text = _non_empty_string(self.name, "old_text", arguments["old_text"])
        new_text = _string(self.name, "new_text", arguments["new_text"])
        try:
            info = await context.workspace.inspect_path(path)
            self.config.authorize(info.canonical_path, write=True)
            _require_file(info, path)
            if info.size > self.config.max_file_bytes:
                raise ValueError(
                    f"File {path!r} is too large ({info.size} bytes; limit {self.config.max_file_bytes})"
                )
            result = await context.workspace.read_file(path)
            if result.is_binary:
                raise ValueError(f"File {path!r} is binary and cannot be edited as text")
            count = result.content.count(old_text)
            if count == 0:
                raise ValueError(f"old_text was not found in {path!r}")
            if count > 1:
                raise ValueError(f"old_text occurs {count} times in {path!r}; include more context")
            content = result.content.replace(old_text, new_text, 1)
            _check_content_size(path, content, self.config.max_file_bytes)
            await context.workspace.write_file(path, content)
            return ToolResult(content=f"Edited {path}.")
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)


@dataclass(frozen=True, slots=True)
class ListDirectoryTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "list_directory"
    description: str = "List visible, authorized entries in a workspace directory."

    def definition(self) -> ToolDefinition:
        return _definition(self, {"path": {"type": "string"}})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(self.name, arguments, required=set(), optional={"path"})
        path = _non_empty_string(self.name, "path", arguments.get("path", "."))
        try:
            root = await context.workspace.inspect_path(path)
            self.config.authorize(root.canonical_path, check_allowed=False)
            _require_directory(root, path)
            entries = await context.workspace.list_directory(path)
            rendered = [
                f"{entry.path}/" if entry.is_directory else f"{entry.path}  ({entry.size} bytes)"
                for entry in entries
                if _is_visible(entry.path) and self.config.is_accessible(entry.canonical_path)
            ]
            return ToolResult(content=_bounded_lines(rendered, self.config.max_list_results, "entries"))
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)


@dataclass(frozen=True, slots=True)
class FindFilesTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "find_files"
    description: str = "Find authorized workspace paths by relative glob pattern."

    def definition(self) -> ToolDefinition:
        return _definition(
            self,
            {"pattern": {"type": "string"}, "path": {"type": "string"}},
            required=["pattern"],
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(self.name, arguments, required={"pattern"}, optional={"path"})
        pattern = _non_empty_string(self.name, "pattern", arguments["pattern"])
        path = _non_empty_string(self.name, "path", arguments.get("path", "."))
        absolute_pattern = (
            PurePath(pattern).is_absolute()
            or PurePosixPath(pattern).is_absolute()
            or PureWindowsPath(pattern).is_absolute()
        )
        if absolute_pattern:
            raise ToolArgumentsError("find_files pattern must be relative")
        if ".." in PurePosixPath(pattern).parts:
            raise ToolArgumentsError("find_files pattern cannot contain '..'")
        try:
            root = await context.workspace.inspect_path(path)
            self.config.authorize(root.canonical_path, check_allowed=False)
            _require_directory(root, path)
            entries = await context.workspace.list_directory(path, recursive=True)
            matches = [
                f"{entry.path}/" if entry.is_directory else entry.path
                for entry in entries
                if _is_visible(entry.path)
                and self.config.is_accessible(entry.canonical_path)
                and _glob_matches(_relative_to_search_root(entry.path, root.path), pattern)
            ]
            return ToolResult(content=_bounded_lines(matches, self.config.max_find_results, "matches"))
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)


@dataclass(frozen=True, slots=True)
class SearchFilesTool:
    config: FileSystemConfig = FileSystemConfig()
    name: str = "search_files"
    description: str = "Search bounded UTF-8 workspace files using a regular expression."

    def definition(self) -> ToolDefinition:
        return _definition(
            self,
            {
                "pattern": {"type": "string"},
                "path": {"type": "string"},
                "include_glob": {"type": "string"},
            },
            required=["pattern"],
        )

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        _validate_arguments(
            self.name,
            arguments,
            required={"pattern"},
            optional={"path", "include_glob"},
        )
        pattern = _string(self.name, "pattern", arguments["pattern"])
        path = _non_empty_string(self.name, "path", arguments.get("path", "."))
        include_glob = arguments.get("include_glob")
        if include_glob is not None:
            include_glob = _non_empty_string(self.name, "include_glob", include_glob)
        try:
            compiled = re.compile(pattern)
        except re.error as error:
            return _error_result(ValueError(f"Invalid regex pattern: {error}"))

        try:
            root = await context.workspace.inspect_path(path)
            self.config.authorize(root.canonical_path, check_allowed=False)
            if not root.exists:
                raise FileNotFoundError(f"Workspace path not found: {path!r}")
            entries = await context.workspace.list_directory(path, recursive=True) if root.is_directory else [root]
            matches: list[str] = []
            truncated = False
            for entry in entries:
                if entry.is_directory or not _is_visible(entry.path):
                    continue
                if not self.config.is_accessible(entry.canonical_path):
                    continue
                if include_glob is not None and not _matches(entry.path, include_glob):
                    continue
                if entry.size > self.config.max_file_bytes:
                    continue
                try:
                    result = await context.workspace.read_file(entry.path)
                except _RECOVERABLE_ERRORS:
                    continue
                if result.is_binary:
                    continue
                for line_number, line in enumerate(result.content.splitlines(), start=1):
                    if not compiled.search(line):
                        continue
                    if len(matches) >= self.config.max_search_results:
                        truncated = True
                        break
                    matches.append(f"{entry.path}:{line_number}:{line}")
                if truncated:
                    break
            if truncated:
                matches.append(f"[... truncated at {self.config.max_search_results} matches]")
            return ToolResult(content="\n".join(matches) if matches else "No matches found.")
        except _RECOVERABLE_ERRORS as error:
            return _error_result(error)

def filesystem_tools(config: FileSystemConfig | None = None) -> list[Tool]:
    config = config or FileSystemConfig()
    tools: list[Tool] = [
        ReadFileTool(config),
        WriteFileTool(config),
        EditFileTool(config),
        ListDirectoryTool(config),
        FindFilesTool(config),
        SearchFilesTool(config),
    ]
    if config.read_only:
        return [tool for tool in tools if tool.name not in {"write_file", "edit_file"}]
    return tools

def _definition(
    tool: Tool,
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> ToolDefinition:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = required
    return ToolDefinition(name=tool.name, description=tool.description, parameters=parameters)


def _validate_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
) -> None:
    supplied = set(arguments)
    missing = required - supplied
    unexpected = supplied - required - optional
    if missing:
        raise ToolArgumentsError(f"Tool {tool_name!r} is missing required argument(s): {', '.join(sorted(missing))}")
    if unexpected:
        raise ToolArgumentsError(f"Tool {tool_name!r} received unexpected argument(s): {', '.join(sorted(unexpected))}")

def _string(tool_name: str, argument_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} must be a string")
    return value


def _non_empty_string(tool_name: str, argument_name: str, value: Any) -> str:
    value = _string(tool_name, argument_name, value)
    if not value.strip():
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} cannot be blank")
    return value

def _positive_int(tool_name: str, argument_name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} must be a positive integer")
    return value


def _non_negative_int(tool_name: str, argument_name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} must be a non-negative integer")
    return value


def _require_file(info: FileInfo, path: str) -> None:
    if not info.exists:
        raise FileNotFoundError(f"Workspace file not found: {path!r}")
    if info.is_directory:
        raise IsADirectoryError(f"Workspace path is a directory: {path!r}")


def _require_directory(info: FileInfo, path: str) -> None:
    if not info.exists:
        raise FileNotFoundError(f"Workspace path not found: {path!r}")
    if not info.is_directory:
        raise NotADirectoryError(f"Workspace path is not a directory: {path!r}")

def _check_content_size(path: str, content: str, maximum: int) -> None:
    size = len(content.encode("utf-8"))
    if size > maximum:
        raise ValueError(f"Content for {path!r} is too large ({size} bytes; limit {maximum})")

def _format_lines(lines: list[str], offset: int, limit: int) -> str:
    if not lines:
        if offset:
            raise ValueError(f"Offset {offset} exceeds empty file length")
        return "(empty file)\n"
    if offset >= len(lines):
        raise ValueError(f"Offset {offset} exceeds file length ({len(lines)} lines)")
    selected = lines[offset : offset + limit]
    rendered = "".join(f"{number:>6}\t{line}" for number, line in enumerate(selected, start=offset + 1))
    if not rendered.endswith("\n"):
        rendered += "\n"
    remaining = len(lines) - offset - len(selected)
    if remaining:
        rendered += f"... ({remaining} more lines. Use offset={offset + len(selected)} to continue.)\n"
    return rendered

def _bounded_lines(lines: list[str], limit: int, noun: str) -> str:
    if not lines:
        return "No matches found." if noun == "matches" else "(empty directory)"
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join([*lines[:limit], f"[... truncated at {limit} {noun}]"])


def _first_match(path: str, patterns: tuple[str, ...]) -> str | None:
    return next((pattern for pattern in patterns if _matches(path, pattern)), None)


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatch(path, pattern) or (pattern.startswith("**/") and fnmatch.fnmatch(path, pattern[3:]))


def _is_visible(path: str) -> bool:
    return not any(part.startswith(".") for part in PurePosixPath(path).parts)


def _relative_to_search_root(path: str, root: str) -> str:
    if root == ".":
        return path
    prefix = root.rstrip("/") + "/"
    return path[len(prefix) :] if path.startswith(prefix) else path


def _glob_matches(path: str, pattern: str) -> bool:
    if "/" not in pattern:
        return "/" not in path and fnmatch.fnmatchcase(path, pattern)
    candidate = PurePosixPath(path)
    return candidate.match(pattern) or (pattern.startswith("**/") and candidate.match(pattern[3:]))

def _error_result(error: BaseException) -> ToolResult:
    return ToolResult(content=str(error), is_error=True)