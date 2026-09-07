"""Docker-backed disposable workspace."""

import asyncio
import posixpath
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from codekeel.workspace.local import LocalWorkspace
from codekeel.workspace.models import CommandResult, FileInfo, FileResult

_DOCKER = "docker"
_LIFECYCLE_TIMEOUT = 120.0


class DockerWorkspace:
    """Execute commands inside a resource-limited disposable container."""

    def __init__(
        self,
        root: str | Path,
        *,
        image: str,
        working_dir: str = "/workspace",
        cpu_limit: float = 1.0,
        memory_limit: str = "512m",
        network_enabled: bool = False,
        command_timeout: float = 30.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.image = image
        self.working_dir = working_dir
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self.network_enabled = network_enabled
        self.command_timeout = command_timeout
        self._container_id: str | None = None
        self._started = False
        self._closed = False
        self._validate_config()
        self._file_workspace = LocalWorkspace(self.root)

    @property
    def container_id(self) -> str | None:
        """Return the managed container ID, if one has been created."""
        return self._container_id

    async def create(self) -> str:
        """Create the container without starting it."""
        self._ensure_open()
        if self._container_id is not None:
            return self._container_id

        result = await self._run_cli(self._create_command(), timeout=_LIFECYCLE_TIMEOUT)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(f"Failed to create Docker workspace: {result.stderr.strip()}")
        self._container_id = result.stdout.strip()
        return self._container_id

    async def start(self) -> None:
        """Create and start the managed container."""
        self._ensure_open()
        container_id = await self.create()
        if self._started:
            return

        result = await self._run_cli(
            [_DOCKER, "start", container_id],
            timeout=_LIFECYCLE_TIMEOUT,
        )
        if result.returncode != 0:
            await self.remove()
            raise RuntimeError(f"Failed to start Docker workspace: {result.stderr.strip()}")
        self._started = True

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        inherit_env: bool = True,
    ) -> CommandResult:
        """Execute a shell command inside the container."""
        return await self._execute_argv(
            ["sh", "-lc", command],
            cwd=self._container_path(cwd or self.working_dir),
            env=env,
            timeout=self.command_timeout if timeout is None else timeout,
            inherit_env=inherit_env,
        )

    async def read_file(self, path: str | Path) -> FileResult:
        """Read a UTF-8 text file through the container boundary."""
        info = await self.inspect_path(path)
        if not info.exists:
            raise FileNotFoundError(f"Workspace path not found: {str(path)!r}")
        if info.is_directory:
            raise IsADirectoryError(f"Workspace path is a directory: {str(path)!r}")
        container_path = self._container_path(info.canonical_path)
        result = await self._execute_argv(
            ["cat", "--", container_path],
            cwd=self.working_dir,
            env=None,
            timeout=self.command_timeout,
        )
        if result.timed_out:
            raise TimeoutError(f"Timed out reading workspace file: {path}")
        if result.exit_code != 0:
            raise FileNotFoundError(f"Unable to read workspace file {path}: {result.stderr.strip()}")
        is_binary = "\x00" in result.stdout[:8192]
        return FileResult(path=str(path), content="" if is_binary else result.stdout, is_binary=is_binary)

    async def write_file(self, path: str | Path, content: str) -> FileResult:
        """Write a UTF-8 text file through the container boundary."""
        info = await self.inspect_path(path)
        if info.exists and info.is_directory:
            raise IsADirectoryError(f"Workspace path is a directory: {str(path)!r}")
        container_path = self._container_path(info.canonical_path)
        parent = str(PurePosixPath(container_path).parent)
        result = await self._execute_argv(
            [
                "sh",
                "-c",
                'mkdir -p -- "$1" && cat > "$2"',
                "write-file",
                parent,
                container_path,
            ],
            cwd=self.working_dir,
            env=None,
            timeout=self.command_timeout,
            input_text=content,
        )
        if result.timed_out:
            raise TimeoutError(f"Timed out writing workspace file: {path}")
        if result.exit_code != 0:
            raise OSError(f"Unable to write workspace file {path}: {result.stderr.strip()}")
        return FileResult(path=str(path), content=content)

    async def inspect_path(self, path: str | Path) -> FileInfo:
        """Resolve mounted paths through the contained host-side workspace view."""
        return await self._file_workspace.inspect_path(path)

    async def list_directory(self, path: str | Path, *, recursive: bool = False) -> list[FileInfo]:
        """List mounted entries without exposing paths outside the bind root."""
        return await self._file_workspace.list_directory(path, recursive=recursive)

    async def stop(self) -> None:
        """Stop the container while retaining it for a later start."""
        if self._container_id is None or not self._started:
            return
        result = await self._run_cli(
            [_DOCKER, "stop", self._container_id],
            timeout=_LIFECYCLE_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to stop Docker workspace: {result.stderr.strip()}")
        self._started = False

    async def remove(self) -> None:
        """Force-remove the managed container."""
        if self._container_id is None:
            return
        container_id = self._container_id
        result = await self._run_cli(
            [_DOCKER, "rm", "-f", container_id],
            timeout=_LIFECYCLE_TIMEOUT,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to remove Docker workspace: {result.stderr.strip()}")
        self._container_id = None
        self._started = False

    async def close(self) -> None:
        """Stop and remove the container; repeated calls are safe."""
        if self._closed:
            return
        try:
            await self.stop()
        finally:
            await self.remove()
            self._closed = True

    async def __aenter__(self) -> "DockerWorkspace":
        await self.start()
        return self

    async def __aexit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        await self.close()

    def _create_command(self) -> list[str]:
        name = f"codekeel-{uuid.uuid4().hex[:12]}"
        mount = f"type=bind,source={self.root},target={self.working_dir}"
        command = [
            _DOCKER,
            "create",
            "--name",
            name,
            "--init",
            "--workdir",
            self.working_dir,
            "--cpus",
            str(self.cpu_limit),
            "--memory",
            self.memory_limit,
            "--mount",
            mount,
        ]
        if not self.network_enabled:
            command.extend(["--network", "none"])
        command.extend(
            [
                self.image,
                "sh",
                "-c",
                "while :; do sleep 3600; done",
            ]
        )
        return command

    async def _execute_argv(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        env: Mapping[str, str] | None,
        timeout: float | None,
        input_text: str | None = None,
        inherit_env: bool = True,
    ) -> CommandResult:
        await self.start()
        assert self._container_id is not None
        args = [_DOCKER, "exec"]
        if input_text is not None:
            args.append("-i")
        args.extend(["--workdir", cwd])
        if inherit_env:
            for key, value in (env or {}).items():
                args.extend(["--env", f"{key}={value}"])
            process_command = list(command)
        else:
            process_command = ["env", "-i", *(f"{key}={value}" for key, value in (env or {}).items()), *command]
        args.extend([self._container_id, *process_command])

        try:
            result = await self._run_cli(args, timeout=timeout, input_text=input_text)
        except subprocess.TimeoutExpired as error:
            stdout = self._timeout_text(error.stdout)
            stderr = self._timeout_text(error.stderr)
            await self._restart_after_timeout()
            return CommandResult(stdout=stdout, stderr=stderr, exit_code=-1, timed_out=True)
        return CommandResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
        )

    async def _restart_after_timeout(self) -> None:
        assert self._container_id is not None
        container_id = self._container_id
        killed = await self._run_cli(
            [_DOCKER, "kill", container_id],
            timeout=_LIFECYCLE_TIMEOUT,
        )
        if killed.returncode != 0:
            await self.remove()
            raise RuntimeError(f"Failed to kill timed-out Docker workspace: {killed.stderr.strip()}")
        self._started = False
        restarted = await self._run_cli(
            [_DOCKER, "start", container_id],
            timeout=_LIFECYCLE_TIMEOUT,
        )
        if restarted.returncode != 0:
            await self.remove()
            raise RuntimeError(f"Failed to restart Docker workspace: {restarted.stderr.strip()}")
        self._started = True

    async def _run_cli(
        self,
        args: Sequence[str],
        *,
        timeout: float | None,
        input_text: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return await asyncio.to_thread(
            subprocess.run,
            list(args),
            input=input_text,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )

    def _container_path(self, path: str | Path) -> str:
        candidate = PurePosixPath(str(path))
        if not candidate.is_absolute():
            candidate = PurePosixPath(self.working_dir) / candidate
        candidate = PurePosixPath(posixpath.normpath(str(candidate)))
        root = PurePosixPath(self.working_dir)
        if not candidate.is_relative_to(root):
            raise PermissionError(f"Workspace path {str(path)!r} resolves outside the workspace")
        return str(candidate)

    def _validate_config(self) -> None:
        if not self.root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {self.root}")
        if self.root == Path(self.root.anchor):
            raise ValueError("Refusing to mount the host filesystem root")
        for sensitive in (Path.home() / ".ssh", Path.home() / ".aws"):
            if sensitive.resolve(strict=False).is_relative_to(self.root):
                raise ValueError(f"Workspace root would expose sensitive directory: {sensitive}")
        docker_socket = Path("/var/run/docker.sock").resolve(strict=False)
        if docker_socket.is_relative_to(self.root):
            raise ValueError("Workspace root would expose the Docker socket")
        if not self.image.strip():
            raise ValueError("Docker image cannot be blank")
        if self.image.startswith("-"):
            raise ValueError("Docker image cannot be interpreted as an option")
        working_dir = PurePosixPath(self.working_dir)
        if not working_dir.is_absolute() or working_dir == PurePosixPath("/"):
            raise ValueError("Docker working_dir must be an absolute non-root path")
        if "," in str(self.root) or "," in self.working_dir:
            raise ValueError("Docker mount paths cannot contain commas")
        if self.cpu_limit <= 0:
            raise ValueError("cpu_limit must be greater than zero")
        if not self.memory_limit.strip():
            raise ValueError("memory_limit cannot be blank")
        if self.command_timeout <= 0:
            raise ValueError("command_timeout must be greater than zero")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Docker workspace is closed")

    @staticmethod
    def _timeout_text(value: str | bytes | None) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value or ""