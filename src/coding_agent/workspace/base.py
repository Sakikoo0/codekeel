"""Provider-independent workspace contract."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from coding_agent.workspace.models import CommandResult, FileInfo, FileResult


class Workspace(Protocol):
    """Execution and file-operation boundary used by the agent runtime."""

    async def execute(
        self, 
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        inherit_env: bool = True
    ) -> CommandResult:
        """Execute with a contained cwd, optional environment, and timeout."""
        ...

    async def read_file(self, path: str | Path) -> FileResult:
        """Read a UTF-8 text file from the workspace."""
        ...

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        """Write a UTF-8 text file in the workspace."""
        ...

    async def inspect_path(self, path: str | Path) -> FileInfo:
        """Resolve a relative path and return contained canonical metadata."""
        ...

    async def list_directory(self, path: str | Path, *, recursive: bool = False) -> list[FileInfo]:
        """List contained entries below a relative workspace directory."""
        ...

    async def close(self) -> None:
        """Release resources owned by the workspace."""
        ...