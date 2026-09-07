"""Workspace implementation backed by the local host."""

import asyncio
import os
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

from codekeel.workspace.models import CommandResult, FileInfo, FileResult


class LocalWorkspace:
    """Run commands and file operations directly on a trusted local host."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {self.root}")

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        inherit_env: bool = True,
    ) -> CommandResult:
        """Execute a command, defaulting to the root and inherited environment.

        Relative working directories are resolved from ``root``. Environment
        values supplied by the caller override inherited host values unless
        inheritance is explicitly disabled.
        """
        return await asyncio.to_thread(
            self._execute,
            command,
            cwd=cwd,
            env=env,
            timeout=timeout,
            inherit_env=inherit_env,
        )

    def _execute(
        self,
        command: str,
        *,
        cwd: str | Path | None,
        env: Mapping[str, str] | None,
        timeout: float | None,
        inherit_env: bool,
    ) -> CommandResult:
        working_dir = self._workspace_path("." if cwd is None else cwd)
        if not working_dir.exists():
            raise FileNotFoundError("The working directory does not exist")
        if not working_dir.is_dir():
            raise NotADirectoryError("The working directory is not a directory")
        process_env = (dict(os.environ) if inherit_env else {}) | dict(env or {})
        process = subprocess.Popen(
            command,
            shell=True,
            cwd=working_dir,
            env=process_env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            stdout, stderr = process.communicate()
            return CommandResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=-1,
                timed_out=True,
            )
        return CommandResult(
            stdout=stdout,
            stderr=stderr,
            exit_code=process.returncode,
        )

    async def read_file(self, path: str | Path) -> FileResult:
        workspace_path = self._workspace_path(path)
        try:
            data = await asyncio.to_thread(workspace_path.read_bytes)
        except OSError as error:
            raise self._safe_file_error(error, path, operation="read") from error
        if b"\x00" in data[:8192]:
            return FileResult(path=str(path), content="", is_binary=True)
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as error:
            raise UnicodeError(f"Workspace file {str(path)!r} is not valid UTF-8") from error
        return FileResult(path=str(path), content=content)

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        workspace_path = self._workspace_path(path)
        try:
            await asyncio.to_thread(self._write_text, workspace_path, content)
        except OSError as error:
            raise self._safe_file_error(error, path, operation="write") from error
        return FileResult(path=str(path), content=content)

    async def inspect_path(self, path: str | Path) -> FileInfo:
        """Return canonical metadata without exposing the host root."""
        return await asyncio.to_thread(self._inspect_path, path)

    async def list_directory(self, path: str | Path, *, recursive: bool = False) -> list[FileInfo]:
        """List entries whose resolved targets remain inside the workspace."""
        return await asyncio.to_thread(self._list_directory, path, recursive=recursive)

    async def close(self) -> None:
        """Release workspace resources; local one-shot operations own none."""

    def _workspace_path(self, path: str | Path) -> Path:
        requested = Path(path)
        if requested.is_absolute():
            raise PermissionError("Workspace path must be relative; absolute paths are not allowed")
        try:
            candidate = (self.root / requested).resolve(strict=False)
        except RuntimeError as error:
            raise PermissionError(f"Workspace path {str(path)!r} resolves through a symlink loop") from error
        except OSError as error:
            reason = error.strerror or type(error).__name__
            raise OSError(f"Unable to resolve workspace path {str(path)!r}: {reason}") from error
        if not candidate.is_relative_to(self.root):
            raise PermissionError(f"Workspace path {str(path)!r} resolves outside the workspace")
        return candidate

    def _inspect_path(self, path: str | Path) -> FileInfo:
        resolved = self._workspace_path(path)
        logical_path = self._logical_path(path)
        exists = resolved.exists()
        if not exists:
            return FileInfo(path=logical_path, canonical_path=self._relative_path(resolved), exists=False)
        try:
            stat = resolved.stat()
        except OSError as error:
            raise self._safe_file_error(error, path, operation="inspect") from error
        return FileInfo(
            path=logical_path,
            canonical_path=self._relative_path(resolved),
            exists=True,
            is_directory=resolved.is_dir(),
            size=stat.st_size,
        )

    def _list_directory(self, path: str | Path, *, recursive: bool) -> list[FileInfo]:
        resolved = self._workspace_path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Workspace path not found: {str(path)!r}")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Workspace path is not a directory: {str(path)!r}")

        entries: list[FileInfo] = []
        iterator = resolved.rglob("*") if recursive else resolved.iterdir()
        try:
            candidates = sorted(iterator, key=lambda entry: entry.as_posix())
        except OSError as error:
            raise self._safe_file_error(error, path, operation="list") from error
        for entry in candidates:
            alias = entry.relative_to(self.root).as_posix()
            try:
                info = self._inspect_path(alias)
            except (OSError, PermissionError, RuntimeError, ValueError):
                continue
            if info.exists:
                entries.append(info)
        return entries

    def _relative_path(self, path: Path) -> str:
        relative = path.relative_to(self.root).as_posix()
        return relative or "."

    @staticmethod
    def _logical_path(path: str | Path) -> str:
        normalized = os.path.normpath(str(path)).replace(os.sep, "/")
        return normalized or "."

    @staticmethod
    def _safe_file_error(error: OSError, path: str | Path, *, operation: str) -> OSError:
        reason = error.strerror or type(error).__name__
        if isinstance(error, PermissionError):
            return PermissionError(f"Unable to {operation} workspace path {str(path)!r}: {reason}")
        if isinstance(error, FileNotFoundError):
            return FileNotFoundError(f"Workspace path not found: {str(path)!r}")
        if isinstance(error, IsADirectoryError):
            return IsADirectoryError(f"Workspace path is a directory: {str(path)!r}")
        if isinstance(error, NotADirectoryError):
            return NotADirectoryError(f"Workspace path is not a directory: {str(path)!r}")
        return OSError(f"Unable to {operation} workspace path {str(path)!r}: {reason}")

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str]) -> None:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass