import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from codekeel.agent import Agent, AgentProtocolError, RunStatus
from codekeel.models import FakeModel, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from codekeel.tools import ToolContext, ToolRegistry
from codekeel.workspace import CommandResult, FileInfo, FileResult


class FakeWorkspace:
    def __init__(self, files: dict[str, str] | None = None) -> None:
        self.commands: list[str] = []
        self.reads: list[str] = []
        self.writes: list[tuple[str, str]] = []
        self.files = dict(files or {})

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
        return CommandResult(stdout="fake output\n", stderr="", exit_code=0)

    async def read_file(self, path: str | Path) -> FileResult:
        logical_path = str(path)
        self.reads.append(logical_path)
        try:
            content = self.files[logical_path]
        except KeyError as error:
            raise FileNotFoundError(logical_path) from error
        return FileResult(path=logical_path, content=content)

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        logical_path = str(path)
        self.writes.append((logical_path, content))
        self.files[logical_path] = content
        return FileResult(path=logical_path, content=content)

    async def inspect_path(self, path: str | Path) -> FileInfo:
        logical_path = str(path)
        content = self.files.get(logical_path)
        return FileInfo(
            path=logical_path,
            canonical_path=logical_path,
            exists=content is not None,
            size=len(content.encode()) if content is not None else 0,
        )

    async def list_directory(self, path: str | Path, *, recursive: bool = False) -> list[FileInfo]:
        return []

    async def close(self) -> None:
        pass


class ContextRecordingTool:
    name = "inspect_context"
    description = "Record the tool context."

    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, Any], ToolContext]] = []

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description, parameters={"type": "object"})

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.calls.append((arguments, context))
        return ToolResult(content="context recorded")


async def test_agent_dispatches_read_write_shell_then_completes() -> None:
    workspace = FakeWorkspace({"input.txt": "hello\n"})
    model = FakeModel(
        [
            ModelResponse(
                tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"path": "input.txt"})],
                usage=Usage(input_tokens=10, output_tokens=4, cost=0.01),
            ),
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="write_file",
                        arguments={"path": "output.txt", "content": "updated\n"},
                    )
                ],
                usage=Usage(input_tokens=5, output_tokens=3, cost=0.01),
            ),
            ModelResponse(
                tool_calls=[ToolCall(id="call-3", name="shell", arguments={"command": "pytest"})],
                usage=Usage(input_tokens=6, output_tokens=2, cost=0.01),
            ),
            ModelResponse(
                content="done",
                usage=Usage(input_tokens=12, output_tokens=2, cost=0.02),
            ),
        ]
    )
    agent = Agent(model=model, workspace=workspace)

    state = await agent.run("Update the file and run tests")

    assert state.status is RunStatus.COMPLETED
    assert [message.role for message in state.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
        "tool",
        "assistant",
    ]
    read_result = json.loads(state.messages[3].content or "")
    assert "hello" in read_result["content"]
    assert read_result["is_error"] is False
    write_result = json.loads(state.messages[5].content or "")
    assert write_result["content"] == "Wrote 8 characters (1 lines) to output.txt."
    assert write_result["is_error"] is False
    shell_result = json.loads(state.messages[7].content or "")
    assert shell_result["content"] == "[stdout]\nfake output\n\n[exit code: 0]"
    assert shell_result["is_error"] is False
    assert [state.messages[index].tool_call_id for index in (3, 5, 7)] == ["call-1", "call-2", "call-3"]
    assert state.messages[-1].content == "done"
    assert state.usage == Usage(input_tokens=33, output_tokens=11, cost=0.05)
    assert workspace.reads == ["input.txt"]
    assert workspace.writes == [("output.txt", "updated\n")]
    assert workspace.commands == ["pytest"]
    assert workspace.files["output.txt"] == "updated\n"


async def test_agent_rejects_unknown_tool_without_execution() -> None:
    workspace = FakeWorkspace()
    agent = Agent(
        model=FakeModel(
            [
                ModelResponse(
                    tool_calls=[ToolCall(id="call-1", name="delete_everything", arguments={})],
                )
            ]
        ),
        workspace=workspace,
    )

    with pytest.raises(AgentProtocolError, match="Unsupported tool"):
        await agent.run("Use an unsupported tool")

    assert agent.state.status is RunStatus.FAILED
    assert workspace.commands == []

async def test_agent_rejects_shell_call_without_command() -> None:
    workspace = FakeWorkspace()
    agent = Agent(
        model=FakeModel([ModelResponse(tool_calls=[ToolCall(id="call-1", name="shell", arguments={})])]),
        workspace=workspace,
    )

    with pytest.raises(AgentProtocolError, match="missing required argument.*command"):
        await agent.run("Run an invalid shell call")

    assert agent.state.status is RunStatus.FAILED
    assert workspace.commands == []


async def test_agent_returns_missing_file_error_to_model_and_continues() -> None:
    workspace = FakeWorkspace()
    agent = Agent(
        model=FakeModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"path": "missing.txt"})]),
                ModelResponse(content="The file does not exist."),
            ]
        ),
        workspace=workspace,
    )

    state = await agent.run("Read a missing file")

    assert state.status is RunStatus.COMPLETED
    tool_result = json.loads(state.messages[3].content or "")
    assert tool_result["is_error"] is True
    assert "missing.txt" in tool_result["content"]
    assert workspace.reads == []


async def test_agent_supplies_workspace_and_run_id_to_tool_context() -> None:
    workspace = FakeWorkspace()
    tool = ContextRecordingTool()
    agent = Agent(
        model=FakeModel(
            [
                ModelResponse(tool_calls=[ToolCall(id="call-1", name=tool.name, arguments={"value": 7})]),
                ModelResponse(content="done"),
            ]
        ),
        workspace=workspace,
        tool_registry=ToolRegistry([tool]),
    )

    await agent.run("Inspect the runtime context")

    assert len(tool.calls) == 1
    arguments, context = tool.calls[0]
    assert arguments == {"value": 7}
    assert context.workspace is workspace
    assert context.run_id