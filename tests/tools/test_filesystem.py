from pathlib import Path

import pytest

from codekeel.models import ToolResult
from codekeel.tools import (
    EditFileTool,
    FileSystemConfig,
    FindFilesTool,
    ListDirectoryTool,
    ReadFileTool,
    SearchFilesTool,
    ToolArgumentsError,
    ToolContext,
    WriteFileTool,
    filesystem_tools,
)
from codekeel.workspace import LocalWorkspace


@pytest.fixture
def workspace_root(tmp_path: Path) -> Path:
    (tmp_path / "hello.txt").write_text("hello\nworld\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hello')\n", encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "config").write_text("private\n", encoding="utf-8")
    (tmp_path / ".env").write_text("TOKEN=secret\n", encoding="utf-8")
    return tmp_path


def context(root: Path) -> ToolContext:
    return ToolContext(workspace=LocalWorkspace(root), run_id="run-filesystem")


async def test_filesystem_tools_happy_path(workspace_root: Path) -> None:
    ctx = context(workspace_root)

    read = await ReadFileTool().execute({"path": "hello.txt", "limit": 1}, ctx)
    write = await WriteFileTool().execute({"path": "new.txt", "content": "before\n"}, ctx)
    edit = await EditFileTool().execute(
        {"path": "new.txt", "old_text": "before", "new_text": "after"},
        ctx,
    )
    listed = await ListDirectoryTool().execute({"path": "."}, ctx)
    found = await FindFilesTool().execute({"pattern": "**/*.py"}, ctx)
    searched = await SearchFilesTool().execute({"pattern": "hello"}, ctx)

    assert "     1\thello" in read.content
    assert "1 more lines" in read.content
    assert write.is_error is False
    assert edit.is_error is False
    assert (workspace_root / "new.txt").read_text() == "after\n"
    assert "hello.txt" in listed.content
    assert ".git" not in listed.content
    assert found.content == "src/main.py"
    assert "hello.txt:1:hello" in searched.content
    assert "src/main.py:1:print('hello')" in searched.content


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (ReadFileTool(), {"path": "../outside.txt"}),
        (WriteFileTool(), {"path": "../outside.txt", "content": "changed"}),
        (EditFileTool(), {"path": "../outside.txt", "old_text": "outside", "new_text": "changed"}),
        (ListDirectoryTool(), {"path": ".."}),
        (FindFilesTool(), {"pattern": "*", "path": ".."}),
        (SearchFilesTool(), {"pattern": "outside", "path": ".."}),
    ],
)
async def test_all_filesystem_tools_reject_parent_traversal(
    workspace_root: Path,
    tool,
    arguments,
) -> None:
    outside = workspace_root.parent / "outside.txt"
    outside.write_text("outside", encoding="utf-8")

    result = await tool.execute(arguments, context(workspace_root))

    assert result.is_error is True
    assert "outside the workspace" in result.content
    assert outside.read_text() == "outside"


@pytest.mark.parametrize("tool", [ReadFileTool(), WriteFileTool()])
async def test_absolute_paths_are_rejected_without_host_path_leak(workspace_root: Path, tool) -> None:
    absolute = str(workspace_root / "hello.txt")
    arguments = {"path": absolute}
    if isinstance(tool, WriteFileTool):
        arguments["content"] = "changed"

    result = await tool.execute(arguments, context(workspace_root))

    assert result.is_error is True
    assert absolute not in result.content
    assert "must be relative" in result.content


async def test_symlink_escape_is_rejected_and_hidden_from_walkers(workspace_root: Path) -> None:
    outside = workspace_root.parent / "outside-secret.txt"
    outside.write_text("outside secret marker\n", encoding="utf-8")
    (workspace_root / "escape.txt").symlink_to(outside)
    ctx = context(workspace_root)

    read = await ReadFileTool().execute({"path": "escape.txt"}, ctx)
    write = await WriteFileTool().execute({"path": "escape.txt", "content": "changed"}, ctx)
    listed = await ListDirectoryTool().execute({}, ctx)
    found = await FindFilesTool().execute({"pattern": "*.txt"}, ctx)
    searched = await SearchFilesTool().execute({"pattern": "outside secret"}, ctx)

    assert read.is_error is True
    assert write.is_error is True
    assert "escape.txt" not in listed.content
    assert "escape.txt" not in found.content
    assert "outside secret" not in searched.content
    assert outside.read_text() == "outside secret marker\n"


@pytest.mark.parametrize("path", [".git/config", ".env"])
async def test_default_protected_paths_are_readable_but_not_writable(workspace_root: Path, path: str) -> None:
    ctx = context(workspace_root)

    read = await ReadFileTool().execute({"path": path}, ctx)
    write = await WriteFileTool().execute({"path": path, "content": "changed"}, ctx)

    assert read.is_error is False
    assert write.is_error is True
    assert "protected" in write.content
    assert "changed" not in (workspace_root / path).read_text()


async def test_custom_protected_pattern_applies_to_canonical_symlink_target(workspace_root: Path) -> None:
    protected = workspace_root / "config.secret"
    protected.write_text("keep\n", encoding="utf-8")
    (workspace_root / "alias.txt").symlink_to(protected)
    config = FileSystemConfig(protected_patterns=("*.secret",))

    result = await WriteFileTool(config).execute(
        {"path": "alias.txt", "content": "changed\n"},
        context(workspace_root),
    )

    assert result.is_error is True
    assert protected.read_text() == "keep\n"


async def test_allowed_and_denied_patterns_filter_direct_and_walker_access(workspace_root: Path) -> None:
    config = FileSystemConfig(allowed_patterns=("*.py",), denied_patterns=("src/private.py",))
    (workspace_root / "src" / "private.py").write_text("secret_marker\n", encoding="utf-8")
    ctx = context(workspace_root)

    denied_read = await ReadFileTool(config).execute({"path": "src/private.py"}, ctx)
    non_allowed_read = await ReadFileTool(config).execute({"path": "hello.txt"}, ctx)
    listed = await ListDirectoryTool(config).execute({"path": "src"}, ctx)
    searched = await SearchFilesTool(config).execute({"pattern": "secret_marker"}, ctx)

    assert denied_read.is_error is True
    assert non_allowed_read.is_error is True
    assert "main.py" in listed.content
    assert "private.py" not in listed.content
    assert searched.content == "No matches found."


def test_read_only_mode_exposes_no_mutating_tools() -> None:
    tools = filesystem_tools(FileSystemConfig(read_only=True))

    assert [tool.name for tool in tools] == ["read_file", "list_directory", "find_files", "search_files"]


async def test_read_only_config_defends_direct_write_invocation(workspace_root: Path) -> None:
    result = await WriteFileTool(FileSystemConfig(read_only=True)).execute(
        {"path": "new.txt", "content": "changed"},
        context(workspace_root),
    )

    assert result.is_error is True
    assert not (workspace_root / "new.txt").exists()


async def test_binary_invalid_utf8_and_oversized_files_are_bounded(workspace_root: Path) -> None:
    (workspace_root / "binary.bin").write_bytes(b"safe-prefix\x00SECRET")
    invalid_utf8_path = workspace_root / "invalid.txt"
    invalid_utf8_path.write_bytes(b"\xff\xfe")
    (workspace_root / "large.txt").write_text("12345", encoding="utf-8")
    ctx = context(workspace_root)

    binary = await ReadFileTool().execute({"path": "binary.bin"}, ctx)
    invalid = await ReadFileTool().execute({"path": invalid_utf8_path.name}, ctx)
    oversized = await ReadFileTool(FileSystemConfig(max_file_bytes=4)).execute({"path": "large.txt"}, ctx)

    assert binary.is_error is False
    assert "SECRET" not in binary.content
    assert "Binary file" in binary.content
    assert invalid.is_error is True
    assert "not valid UTF-8" in invalid.content
    assert oversized.is_error is True
    assert "too large" in oversized.content


async def test_oversized_write_is_rejected_before_file_creation(workspace_root: Path) -> None:
    result = await WriteFileTool(FileSystemConfig(max_file_bytes=4)).execute(
        {"path": "large-write.txt", "content": "12345"},
        context(workspace_root),
    )

    assert result.is_error is True
    assert "too large" in result.content
    assert not (workspace_root / "large-write.txt").exists()


async def test_nonexistent_paths_return_model_correctable_errors(workspace_root: Path) -> None:
    result = await ReadFileTool().execute({"path": "missing.txt"}, context(workspace_root))

    assert result == ToolResult(content="Workspace file not found: 'missing.txt'", is_error=True)


async def test_edit_requires_exactly_one_match(workspace_root: Path) -> None:
    (workspace_root / "repeat.txt").write_text("same same\n", encoding="utf-8")
    ctx = context(workspace_root)

    absent = await EditFileTool().execute(
        {"path": "repeat.txt", "old_text": "missing", "new_text": "new"},
        ctx,
    )
    ambiguous = await EditFileTool().execute(
        {"path": "repeat.txt", "old_text": "same", "new_text": "new"},
        ctx,
    )

    assert absent.is_error is True
    assert ambiguous.is_error is True
    assert (workspace_root / "repeat.txt").read_text() == "same same\n"


async def test_list_find_and_search_limits_are_explicit(workspace_root: Path) -> None:
    for index in range(3):
        (workspace_root / f"match{index}.txt").write_text("needle\n", encoding="utf-8")
    config = FileSystemConfig(max_list_results=1, max_find_results=1, max_search_results=1)
    ctx = context(workspace_root)

    listed = await ListDirectoryTool(config).execute({}, ctx)
    found = await FindFilesTool(config).execute({"pattern": "*.txt"}, ctx)
    searched = await SearchFilesTool(config).execute({"pattern": "needle"}, ctx)

    assert "truncated at 1 entries" in listed.content
    assert "truncated at 1 matches" in found.content
    assert "truncated at 1 matches" in searched.content


@pytest.mark.parametrize(
    ("tool", "arguments", "returns_error_result"),
    [
        (ReadFileTool(), {"path": "hello.txt", "offset": -1}, False),
        (ReadFileTool(), {"path": "hello.txt", "limit": True}, False),
        (FindFilesTool(), {"pattern": "../*"}, False),
        (SearchFilesTool(), {"pattern": "[invalid"}, True),
        (WriteFileTool(), {"path": "hello.txt", "content": "ok", "extra": True}, False),
    ],
)
async def test_invalid_arguments_never_mutate_workspace(
    workspace_root: Path,
    tool,
    arguments,
    returns_error_result: bool,
) -> None:
    original = (workspace_root / "hello.txt").read_text()

    if returns_error_result:
        result = await tool.execute(arguments, context(workspace_root))
        assert result.is_error is True
    else:
        with pytest.raises(ToolArgumentsError):
            await tool.execute(arguments, context(workspace_root))

    assert (workspace_root / "hello.txt").read_text() == original
