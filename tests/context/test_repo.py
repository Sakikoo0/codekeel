import json
from unittest.mock import AsyncMock

import pytest

from coding_agent.agent import Agent, RunStatus
from coding_agent.context import RepoContextConfig, discover_repo_context
from coding_agent.models import FakeModel, ModelResponse, ToolCall
from coding_agent.workspace import CommandResult, FileInfo, FileResult, LocalWorkspace


class FakeWorkspace:
    def __init__(self, files=None):
        self.files = files or {}
        self.reads = []
        self.execute = AsyncMock(return_value=CommandResult("main\n", "", 0))
        self.list_directory = AsyncMock(return_value=[])

    async def inspect_path(self, path):
        return FileInfo(path, path, path in self.files, size=len(self.files.get(path, "").encode()))

    async def read_file(self, path):
        self.reads.append(path)
        return FileResult(path, self.files[path])


async def test_discovers_only_root_files_and_deduplicates_alias(tmp_path):
    workspace = LocalWorkspace(tmp_path)
    for name in ("AGENTS.md", "README.md", "pyproject.toml", "package.json"):
        await workspace.write_file(name, f"contents of {name}")
    (tmp_path / "CLAUDE.md").symlink_to("AGENTS.md")
    await workspace.write_file("nested/AGENTS.md", "nested instructions")
    context = await discover_repo_context(workspace)
    assert list(context.files) == ["AGENTS.md", "README.md", "pyproject.toml", "package.json"]
    assert "nested instructions" not in context.render()
    assert "nested/" in context.tree


async def test_large_repo_tree_is_sorted_nonrecursive_and_bounded():
    workspace = FakeWorkspace()
    workspace.list_directory.return_value = [
        FileInfo(f"file-{i:04}", f"file-{i:04}", True) for i in reversed(range(10_000))
    ]
    context = await discover_repo_context(workspace, RepoContextConfig(max_tree_entries=3))
    assert context.tree == ("file-0000", "file-0001", "file-0002")
    assert context.tree_truncated
    workspace.list_directory.assert_awaited_once_with(".", recursive=False)
    context = await discover_repo_context(workspace, RepoContextConfig(max_tree_chars=9))
    assert context.tree == ("file-0000",)
    assert context.tree_truncated
    assert workspace.reads == []


async def test_ignored_dirs_and_sensitive_files_are_not_read(tmp_path):
    for name in (".git", ".venv", "node_modules", "vendor", "build", "dist", "__pycache__"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "AGENTS.md").write_text("SECRET")
    for name in (".env", ".env.local", "credentials", "credentials.json", "secrets", "private.key"):
        (tmp_path / name).write_text("SECRET")
    (tmp_path / "README.md").write_text("safe")
    context = await discover_repo_context(LocalWorkspace(tmp_path))
    assert context.tree == ("README.md",)
    assert context.files == {"README.md": "safe"}
    assert "SECRET" not in context.render()


@pytest.mark.parametrize("target", [".env", "credentials.json", "nested/README.md", "../outside.txt"])
async def test_instruction_alias_cannot_read_unapproved_target(tmp_path, target):
    repo = tmp_path / "repo"
    repo.mkdir()
    destination = repo / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("DO_NOT_DISCLOSE")
    (repo / "AGENTS.md").symlink_to(target)
    workspace = LocalWorkspace(repo)
    workspace.read_file = AsyncMock(wraps=workspace.read_file)
    context = await discover_repo_context(workspace)
    workspace.read_file.assert_not_awaited()
    assert "DO_NOT_DISCLOSE" not in context.render()
    assert "AGENTS.md" not in context.files


async def test_oversized_file_is_not_read_and_invalid_text_is_omitted(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 101)
    (tmp_path / "CLAUDE.md").write_bytes(b"\xff")
    (tmp_path / "README.md").write_bytes(b"\x00binary")
    workspace = LocalWorkspace(tmp_path)
    workspace.read_file = AsyncMock(wraps=workspace.read_file)
    context = await discover_repo_context(workspace, RepoContextConfig(max_file_bytes=100))
    assert context.files == {}
    assert [call.args[0] for call in workspace.read_file.await_args_list] == ["CLAUDE.md", "README.md"]
    assert len(context.notices) == 4


async def test_post_read_byte_check_handles_file_growth():
    workspace = FakeWorkspace({"README.md": "ok"})
    workspace.read_file = AsyncMock(return_value=FileResult("README.md", "汉" * 10))
    context = await discover_repo_context(workspace, RepoContextConfig(max_file_bytes=20))
    assert context.files == {}
    assert "README.md: omitted (file byte limit)" in context.notices


async def test_failure_paths_are_partial_and_do_not_expose_error_details():
    workspace = FakeWorkspace({"AGENTS.md": "instructions", "README.md": "readme", ".git": ""})
    workspace.read_file = AsyncMock(side_effect=[FileResult("AGENTS.md", "instructions"), PermissionError("SECRET")])
    workspace.list_directory.side_effect = OSError("SECRET")
    workspace.execute.side_effect = [CommandResult("SECRET", "SECRET", -1, True), OSError("SECRET")]
    context = await discover_repo_context(workspace)
    assert context.files == {"AGENTS.md": "instructions"}
    assert context.git_status is None and context.git_branch is None
    assert context.notices == (
        "README.md: unavailable", "tree: unavailable", "git branch: unavailable", "git status: unavailable",
    )
    assert "SECRET" not in context.render()


async def test_git_output_bounds_and_execution_options():
    workspace = FakeWorkspace({".git": ""})
    workspace.execute.return_value = CommandResult("x" * 100, "SECRET", 0)
    context = await discover_repo_context(workspace, RepoContextConfig(max_git_chars=10))
    assert context.git_branch == context.git_status == "x" * 10
    assert context.git_truncated == ("branch", "status")
    for call in workspace.execute.await_args_list:
        assert call.kwargs["inherit_env"] is False
        assert call.kwargs["timeout"] == 5.0
        assert call.kwargs["cwd"] == "."
        assert "core.fsmonitor=false" in call.args[0]


async def test_real_git_branch_dirty_clean_and_parent_repo(tmp_path):
    workspace = LocalWorkspace(tmp_path)
    for command in (
        "git init -b context-test", "git config user.email test@example.invalid", "git config user.name Test",
        "git -c commit.gpgsign=false commit --allow-empty -m initial",
    ):
        result = await workspace.execute(command)
        assert result.exit_code == 0, result.stderr
    clean = await discover_repo_context(workspace)
    assert clean.git_branch == "context-test"
    assert clean.git_status == ""
    await workspace.write_file("README.md", "dirty")
    dirty = await discover_repo_context(workspace)
    assert dirty.git_status == "?? README.md"
    (tmp_path / "child").mkdir()
    child = LocalWorkspace(tmp_path / "child")
    child.execute = AsyncMock()
    context = await discover_repo_context(child)
    assert context.git_branch is None
    child.execute.assert_not_awaited()


async def test_agent_receives_static_snapshot_and_next_run_does_not_reuse_it():
    workspace = FakeWorkspace({"AGENTS.md": "Use {{literal}} instructions", "CLAUDE.md": "More instructions"})
    context = await discover_repo_context(workspace)
    model = FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="1", name="read_file", arguments={"path": "AGENTS.md"})]),
        ModelResponse(content="done"), ModelResponse(content="second run"),
    ])
    model.complete = AsyncMock(wraps=model.complete)
    agent = Agent(model, workspace)
    state = await agent.run("task", repo_context=context)
    assert state.status is RunStatus.COMPLETED
    assert state.messages[0].content.count("Repository context") == 1
    payload = json.loads(state.messages[0].content.split("untrusted data):\n")[1])
    assert payload["files"] == context.files
    assert state.messages[1].content == "task"
    assert workspace.reads == ["AGENTS.md", "CLAUDE.md", "AGENTS.md"]
    for call in model.complete.await_args_list:
        assert call.args[0][0] == state.messages[0]
    second = await agent.run("another task")
    assert second.messages[0].content == agent.system_prompt


@pytest.mark.parametrize("field", ["max_file_bytes", "max_tree_entries", "max_tree_chars", "max_git_chars"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_limits(field, value):
    with pytest.raises(ValueError, match=field):
        RepoContextConfig(**{field: value})
