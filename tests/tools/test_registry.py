from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from codekeel.models import ToolCall, ToolDefinition, ToolResult
from codekeel.tools import (
    DuplicateToolError,
    ShellTool,
    ToolArgumentsError,
    ToolContext,
    ToolRegistry,
    ToolRegistryError,
    UnknownToolError,
    default_tool_registry,
)
from codekeel.workspace import CommandResult, FileResult


class FakeWorkspace:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.reads: list[str] = []
        self.writes: list[tuple[str, str]] = []

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        inherit_env: bool = True,
    ) -> CommandResult:
        self.commands.append(command)
        return CommandResult(stdout="", stderr="failed\n", exit_code=2)

    async def read_file(self, path: str | Path) -> FileResult:
        self.reads.append(str(path))
        return FileResult(path=str(path), content="content")

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        self.writes.append((str(path), content))
        return FileResult(path=str(path), content=content)

    async def close(self) -> None:
        pass


@dataclass
class RecordingTool:
    name: str = "record"
    description: str = "Record one invocation."
    calls: list[tuple[dict[str, Any], ToolContext]] = field(default_factory=list)

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description, parameters={"type": "object"})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.calls.append((arguments, context))
        return ToolResult(content="recorded")


def test_default_registry_exposes_current_tools_in_stable_order() -> None:
    definitions = default_tool_registry().definitions()

    assert [definition.name for definition in definitions] == [
        "shell",
        "read_file",
        "write_file",
        "edit_file",
        "list_directory",
        "find_files",
        "search_files",
        "update_plan",
    ]
    assert all(definition.parameters["type"] == "object" for definition in definitions)
    assert all(definition.parameters["additionalProperties"] is False for definition in definitions)


async def test_registry_dispatches_tool_call_with_context() -> None:
    workspace = FakeWorkspace()
    context = ToolContext(workspace=workspace, run_id="run-123")
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    call = ToolCall(id="call-1", name="record", arguments={"value": 7})

    result = await registry.execute(call, context)

    assert result == ToolResult(content="recorded")
    assert tool.calls == [({"value": 7}, context)]


def test_registry_rejects_duplicate_name_without_replacing_original() -> None:
    original = RecordingTool()
    registry = ToolRegistry([original])

    with pytest.raises(DuplicateToolError, match="already registered"):
        registry.register(RecordingTool())

    assert registry.get("record") is original


async def test_unknown_tool_is_rejected_without_executing_registered_tools() -> None:
    tool = RecordingTool()
    registry = ToolRegistry([tool])
    context = ToolContext(workspace=FakeWorkspace(), run_id="run-123")

    with pytest.raises(UnknownToolError, match="Unsupported tool"):
        await registry.execute(ToolCall(id="call-1", name="record.extra", arguments={}), context)

    assert tool.calls == []


def test_registry_rejects_tool_with_inconsistent_definition_name() -> None:
    tool = RecordingTool(name="actual")

    def mismatched_definition() -> ToolDefinition:
        return ToolDefinition(name="advertised", description=tool.description)

    tool.definition = mismatched_definition  # type: ignore[method-assign]

    with pytest.raises(ToolRegistryError, match="does not match definition name"):
        ToolRegistry([tool])


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"command": ""},
        {"command": 7},
        {"command": "pytest", "unexpected": True},
    ],
)
async def test_builtin_tool_rejects_invalid_arguments_before_workspace_access(arguments) -> None:
    workspace = FakeWorkspace()
    context = ToolContext(workspace=workspace, run_id="run-123")

    with pytest.raises(ToolArgumentsError):
        await ShellTool().execute(arguments, context)

    assert workspace.commands == []


async def test_shell_tool_marks_nonzero_command_result_as_error() -> None:
    workspace = FakeWorkspace()
    context = ToolContext(workspace=workspace, run_id="run-123")

    result = await ShellTool().execute({"command": "pytest"}, context)

    assert result.is_error is True
    assert "[stderr]\nfailed" in result.content
    assert "[exit code: 2]" in result.content
    assert workspace.commands == ["pytest"]


def test_tool_context_rejects_blank_run_id() -> None:
    with pytest.raises(ValueError, match="run_id cannot be blank"):
        ToolContext(workspace=FakeWorkspace(), run_id=" ")