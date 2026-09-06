from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest

from coding_agent.workspace import CommandResult, FileInfo, FileResult, LocalWorkspace, Workspace


class FakeWorkspace:
    """In-memory workspace used to verify the backend contract."""

    def __init__(self) -> None:
        self.files: dict[str, str] = {}
        self.closed = False

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        inherit_env: bool = True,
    ) -> CommandResult:
        return CommandResult(stdout=f"executed: {command}", stderr="", exit_code=0)

    async def read_file(self, path: str | Path) -> FileResult:
        logical_path = str(path)
        try:
            content = self.files[logical_path]
        except KeyError as error:
            raise FileNotFoundError(logical_path) from error
        return FileResult(path=logical_path, content=content)

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        logical_path = str(path)
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
        prefix = "" if str(path) == "." else f"{path}/"
        return [
            FileInfo(path=name, canonical_path=name, exists=True, size=len(content.encode()))
            for name, content in sorted(self.files.items())
            if name.startswith(prefix) and (recursive or "/" not in name[len(prefix) :])
        ]

    async def close(self) -> None:
        self.closed = True


def _workspace(kind: Literal["local", "fake"], root: Path) -> Workspace:
    if kind == "local":
        return LocalWorkspace(root)
    return FakeWorkspace()


@pytest.mark.parametrize("kind", ["local", "fake"])
async def test_workspace_contract_executes_commands(kind, tmp_path) -> None:
    workspace = _workspace(kind, tmp_path)

    result = await workspace.execute("printf contract")

    assert result.return_code == 0
    if kind == "local":
        assert result.output == "contract"
    else:
        assert result.output == "executed: printf contract"


@pytest.mark.parametrize("kind", ["local", "fake"])
async def test_workspace_contract_writes_then_reads_text(kind, tmp_path) -> None:
    workspace = _workspace(kind, tmp_path)

    written = await workspace.write_file("nested/example.txt", "hello, workspace")
    read = await workspace.read_file("nested/example.txt")

    assert written == FileResult(path="nested/example.txt", content="hello, workspace")
    assert read == written


@pytest.mark.parametrize("kind", ["local", "fake"])
async def test_workspace_contract_reports_missing_files(kind, tmp_path) -> None:
    workspace = _workspace(kind, tmp_path)

    with pytest.raises(FileNotFoundError):
        await workspace.read_file("missing.txt")


@pytest.mark.parametrize("kind", ["local", "fake"])
async def test_workspace_contract_closes(kind, tmp_path) -> None:
    workspace = _workspace(kind, tmp_path)

    await workspace.close()

    if isinstance(workspace, FakeWorkspace):
        assert workspace.closed is True


@pytest.mark.parametrize("kind", ["local", "fake"])
async def test_workspace_contract_inspects_and_lists_files(kind, tmp_path) -> None:
    workspace = _workspace(kind, tmp_path)
    await workspace.write_file("nested/example.txt", "hello")

    info = await workspace.inspect_path("nested/example.txt")
    entries = await workspace.list_directory("nested")

    assert info.exists is True
    assert info.is_directory is False
    assert info.size == 5
    assert entries == [info]
