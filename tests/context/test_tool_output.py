import asyncio
import hashlib
import json
from unittest.mock import AsyncMock

import pytest

from codekeel.agent import Agent, RunStatus
from codekeel.context.tool_output import ToolOutputConfig, ToolOutputManager
from codekeel.models import FakeModel, ModelResponse, ToolCall, ToolDefinition, ToolResult
from codekeel.runtime.policy import ActionPolicy, Risk
from codekeel.tools import ToolRegistry
from codekeel.workspace import CommandResult, FileInfo, FileResult, LocalWorkspace


class FakeWorkspace:
    def __init__(self):
        self.files = {}
        self.writes = []
        self.execute = AsyncMock(return_value=CommandResult("", "", 0))

    async def inspect_path(self, path):
        directory = any(name.startswith(path + "/") for name in self.files)
        return FileInfo(path, path, path in self.files or directory, directory,
                        len(self.files.get(path, "").encode()))

    async def read_file(self, path):
        return FileResult(path, self.files[path])

    async def write_file(self, path, content):
        self.writes.append(path)
        self.files[path] = content
        return FileResult(path, content)


def config(**changes):
    return ToolOutputConfig(**({"max_chars": 1000, "max_lines": 9, "head_lines": 3,
                               "tail_lines": 3, "spill_threshold": 2000} | changes))


async def process(text, workspace=None, **changes):
    return await ToolOutputManager(config(**changes)).process(
        ToolResult(content=text), workspace=workspace or FakeWorkspace(), run_id="test-run",
    )


@pytest.mark.parametrize("text", ["", "tiny\n", "x" * 1000, "a\n" * 9])
async def test_small_results_pass_through_without_workspace_access(text):
    workspace = FakeWorkspace()
    workspace.inspect_path = AsyncMock(side_effect=AssertionError("Unexpected I/O"))
    original = ToolResult(content=text, is_error=True)
    result = await ToolOutputManager(config()).process(original, workspace=workspace, run_id="test")
    assert result is original
    workspace.inspect_path.assert_not_awaited()


@pytest.mark.parametrize("text", ["A" * 1200 + "Z", "first\n" + "middle\n" * 20 + "last"])
async def test_medium_output_keeps_head_tail_and_does_not_spill(text):
    workspace = FakeWorkspace()
    result = await process(text, workspace)
    assert len(result.content) <= 1000
    assert len(result.content.splitlines()) <= 9
    assert result.content.startswith(text[:1])
    assert result.content.endswith(text[-1:])
    assert "Output truncated" in result.content
    assert "not spilled" in result.content
    assert workspace.writes == []


@pytest.mark.parametrize("text", ["HEAD\n" + "中\r\n" * 2000 + "TAIL", "A" + "x" * 5000 + "Z",
                                   "a\u2028" * 3000 + "END"])
async def test_large_result_spills_exact_utf8_content_and_reuses_hash(text):
    workspace = FakeWorkspace()
    manager = ToolOutputManager(config())
    original = ToolResult(content=text, is_error=True)
    result = await manager.process(original, workspace=workspace, run_id="test")
    path = f".agent/runs/test/artifacts/{hashlib.sha256(text.encode()).hexdigest()}.txt"
    assert workspace.files == {path: text}
    assert path in result.content
    assert "sed -n '1,80p'" in result.content
    assert result.is_error
    assert original.content == text
    assert len(result.content) <= 1000
    assert len(result.content.splitlines()) <= 9
    assert result.content.startswith(text[0]) and result.content.endswith(text[-1])
    assert await manager.process(original, workspace=workspace, run_id="test") == result
    assert workspace.writes == [path]
    await manager.process(original, workspace=workspace, run_id="second")
    assert len(workspace.files) == 2


@pytest.mark.parametrize("length,spilled", [(1999, False), (2000, True), (2001, True)])
async def test_spill_threshold_is_inclusive(length, spilled):
    workspace = FakeWorkspace()
    await process("x" * length, workspace)
    assert bool(workspace.writes) is spilled


async def test_smallest_char_budget_preserves_complete_hint_for_longest_run_id():
    text = "A" + "x" * 3000 + "Z"
    workspace = FakeWorkspace()
    result = await ToolOutputManager(config(max_chars=512)).process(
        ToolResult(content=text), workspace=workspace, run_id="a" * 128,
    )
    assert len(result.content) <= 512
    assert workspace.writes[0] in result.content
    assert result.content.startswith("A") and result.content.endswith("Z")


@pytest.mark.parametrize("operation", ["inspect_path", "write_file"])
async def test_spill_failure_falls_back_to_bounded_preview_without_error_details(operation):
    workspace = FakeWorkspace()
    setattr(workspace, operation, AsyncMock(side_effect=PermissionError("SECRET host path")))
    result = await process("A" + "x" * 3000 + "Z", workspace)
    assert "spill failed" in result.content
    assert ".txt" not in result.content and "SECRET" not in result.content
    assert result.content.startswith("A") and result.content.endswith("Z")
    assert len(result.content) <= 1000
    assert not result.is_error


async def test_spill_cancellation_is_not_swallowed():
    workspace = FakeWorkspace()
    workspace.write_file = AsyncMock(side_effect=asyncio.CancelledError())
    with pytest.raises(asyncio.CancelledError):
        await process("x" * 3000, workspace)


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "a/b", "a\\b", "a\n", ""])
async def test_invalid_run_ids_never_touch_workspace(run_id):
    workspace = FakeWorkspace()
    workspace.inspect_path = AsyncMock()
    with pytest.raises(ValueError):
        await ToolOutputManager().process(ToolResult(content="x" * 100_000), workspace=workspace, run_id=run_id)
    workspace.inspect_path.assert_not_awaited()


@pytest.mark.parametrize("component", [".agent", ".agent/runs", ".agent/runs/test-run",
                                       ".agent/runs/test-run/artifacts", "file"])
async def test_symlink_artifacts_cannot_overwrite_other_files(tmp_path, component):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    text = "A" + "x" * 3000 + "Z"
    if component == "file":
        target = outside / "secret"
        target.write_text("SECRET")
        component = f".agent/runs/test-run/artifacts/{hashlib.sha256(text.encode()).hexdigest()}.txt"
    else:
        target = outside
    alias = root / component
    alias.parent.mkdir(parents=True, exist_ok=True)
    alias.symlink_to(target)
    result = await process(text, LocalWorkspace(root))
    assert "spill failed" in result.content
    assert "SECRET" not in result.content
    if target.is_file():
        assert target.read_text() == "SECRET"
    else:
        assert list(outside.iterdir()) == []


async def test_existing_conflicting_artifact_is_not_overwritten():
    workspace = FakeWorkspace()
    text = "x" * 3000
    path = f".agent/runs/test-run/artifacts/{hashlib.sha256(text.encode()).hexdigest()}.txt"
    workspace.files[path] = "SECRET" * 500
    result = await process(text, workspace)
    assert "spill failed" in result.content
    assert workspace.files[path] == "SECRET" * 500
    assert workspace.writes == []


async def test_ten_megabyte_shell_stdout_is_preserved_but_never_enters_model_history():
    stdout = "HEAD\n" + "0123456789\n" * 953250 + "TAIL\n"
    assert len(stdout.encode()) >= 10 * 1024 * 1024
    workspace = FakeWorkspace()
    workspace.execute.return_value = CommandResult(stdout, "", 0)
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="shell-call", name="shell", arguments={"command": "printf big"})]),
        ModelResponse(content="done"),
    ])
    model.complete = AsyncMock(wraps=model.complete)
    agent = Agent(model, workspace)
    state = await agent.run("run")
    assert state.status is RunStatus.COMPLETED
    message = state.messages[3]
    preview = json.loads(message.content)["content"]
    assert message.tool_call_id == "shell-call"
    assert len(preview) <= 10_000 and len(preview.splitlines()) <= 256
    assert "HEAD" in preview and "TAIL" in preview
    assert len(state.model_dump_json()) < 30_000
    expected = f"[stdout]\n{stdout}\n[exit code: 0]"
    assert list(workspace.files.values()) == [expected]
    assert workspace.writes[0] in preview
    events = agent.event_store.read(agent.run_id).events
    complete = next(event for event in events if event.type == "ToolCompleted")
    assert complete.payload.result.content == expected
    requests = [event for event in events if event.type == "ModelRequested"]
    assert len(requests[-1].model_dump_json()) < 30_000
    assert json.loads(model.complete.await_args_list[-1].args[0][3].content)["content"] == preview


async def test_spilled_artifact_is_retrievable_with_existing_shell(tmp_path):
    text = "HEAD\n" + "middle\n" * 1000 + "TAIL"
    workspace = LocalWorkspace(tmp_path)
    result = await process(text, workspace)
    digest = hashlib.sha256(text.encode()).hexdigest()
    path = f".agent/runs/test-run/artifacts/{digest}.txt"
    assert (tmp_path / path).read_text() == text
    assert path in result.content
    output = await workspace.execute(f"sed -n '1,2p' {path}")
    assert output.exit_code == 0 and output.stdout == "HEAD\nmiddle\n"


@pytest.mark.parametrize("name,is_error", [("read_file", False), ("search_files", False), ("failed_tool", True)])
async def test_shared_pipeline_bounds_custom_tool_results_and_preserves_error_semantics(name, is_error):
    original = ToolResult(content="HEAD\n" + "middle\n" * 1000 + "TAIL", is_error=is_error)

    class ResultTool:
        def definition(self):
            return ToolDefinition(name=name, description="Scripted result", parameters={"type": "object"})

        async def execute(self, arguments, context):
            return original

    tool = ResultTool()
    tool.name = name
    workspace = FakeWorkspace()
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="one", name=name, arguments={})]), ModelResponse(content="done"),
    ])
    agent = Agent(model, workspace, policy=ActionPolicy(tool_risks={name: Risk.LOW}),
                  tool_registry=ToolRegistry([tool]), tool_output_manager=ToolOutputManager(config()))
    state = await agent.run("task")
    assert state.status is RunStatus.COMPLETED
    result = json.loads(state.messages[3].content)
    assert result["is_error"] is is_error
    assert len(result["content"]) <= 1000
    assert state.messages[3].tool_call_id == "one"
    assert list(workspace.files.values()) == [original.content]
    events = agent.event_store.read(agent.run_id).events
    completed = next(event for event in events if event.type == ("ToolFailed" if is_error else "ToolCompleted"))
    assert completed.payload.result == original


@pytest.mark.parametrize("field", ["max_chars", "max_lines", "head_lines", "tail_lines", "spill_threshold"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_numeric_config(field, value):
    with pytest.raises(ValueError):
        config(**{field: value})


@pytest.mark.parametrize("changes", [{"max_chars": 511}, {"max_lines": 6}, {"spill_threshold": 1000}])
def test_inconsistent_config_rejected(changes):
    with pytest.raises(ValueError):
        config(**changes)