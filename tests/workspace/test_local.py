import asyncio
import os
import shlex
import signal
import sys
from pathlib import Path

import pytest

from coding_agent.workspace import LocalWorkspace


async def test_local_workspace_runs_each_command_in_a_new_process(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path)

    assert (await workspace.execute("cd ..")).return_code == 0
    result = await workspace.execute("pwd")

    assert result.exit_code == 0
    assert result.stdout.strip() == str(tmp_path.resolve())
    assert result.stderr == ""
    assert result.timed_out is False


async def test_local_workspace_captures_nonzero_exit_and_stderr(tmp_path) -> None:
    result = await LocalWorkspace(tmp_path).execute("printf 'failure\\n' >&2; exit 7")

    assert result.exit_code == 7
    assert result.stdout == ""
    assert result.stderr == "failure\n"
    assert result.timed_out is False


async def test_local_workspace_terminates_timed_out_command(tmp_path) -> None:
    code = "import sys, time; print('before', flush=True); print('error', file=sys.stderr, flush=True); time.sleep(5)"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await LocalWorkspace(tmp_path).execute(command, timeout=0.1)

    assert result.exit_code == -1
    assert result.stdout == "before\n"
    assert result.stderr == "error\n"
    assert result.timed_out is True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_local_workspace_timeout_kills_child_process(tmp_path) -> None:
    pid_file = tmp_path / "child.pid"
    script = tmp_path / "child.py"
    script.write_text(
        "\n".join(
            [
                "import os",
                "import sys",
                "import time",
                "from pathlib import Path",
                "Path(sys.argv[1]).write_text(str(os.getpid()))",
                "while True:",
                "    time.sleep(1)",
            ]
        )
    )
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(script))} {shlex.quote(str(pid_file))}"

    result = await LocalWorkspace(tmp_path).execute(command, timeout=0.5)
    child_pid = await _read_pid(pid_file)

    try:
        assert result.timed_out is True
        assert await _process_exited(child_pid)
    finally:
        _kill_process_if_running(child_pid)


async def test_local_workspace_uses_requested_cwd(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    result = await LocalWorkspace(tmp_path).execute("pwd", cwd="nested")

    assert result.exit_code == 0
    assert result.stdout.strip() == str(nested)


async def test_local_workspace_merges_and_overrides_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HARNESS_INHERITED", "inherited")
    monkeypatch.setenv("HARNESS_OVERRIDE", "host")
    code = (
        "import os; "
        "print(os.environ['HARNESS_INHERITED']); "
        "print(os.environ['HARNESS_OVERRIDE'])"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await LocalWorkspace(tmp_path).execute(
        command,
        env={"HARNESS_OVERRIDE": "workspace"},
    )

    assert result.exit_code == 0
    assert result.stdout.splitlines() == ["inherited", "workspace"]


async def test_local_workspace_can_disable_environment_inheritance(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOST_SECRET", "hidden")
    code = "import os; print(os.environ.get('HOST_SECRET', 'absent')); print(os.environ['SAFE'])"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    result = await LocalWorkspace(tmp_path).execute(
        command,
        env={"SAFE": "visible"},
        inherit_env=False,
    )

    assert result.stdout.splitlines() == ["absent", "visible"]


@pytest.mark.parametrize("cwd", ["..", "../outside", "/tmp"])
async def test_local_workspace_rejects_command_cwd_escape(tmp_path, cwd) -> None:
    with pytest.raises(PermissionError):
        await LocalWorkspace(tmp_path).execute("pwd", cwd=cwd)


@pytest.mark.parametrize("path", ["../outside.txt", "../../outside.txt"])
async def test_local_workspace_file_operations_reject_parent_traversal(tmp_path, path) -> None:
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(PermissionError, match="outside the workspace"):
        await workspace.read_file(path)
    with pytest.raises(PermissionError, match="outside the workspace"):
        await workspace.write_file(path, "changed")


async def test_local_workspace_file_operations_reject_absolute_paths(tmp_path) -> None:
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(PermissionError, match="absolute paths are not allowed"):
        await workspace.inspect_path(tmp_path / "file.txt")


async def test_local_workspace_rejects_symlink_escape_and_omits_it_from_listing(tmp_path) -> None:
    outside = tmp_path.parent / "outside-workspace.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)
    workspace = LocalWorkspace(tmp_path)

    with pytest.raises(PermissionError, match="outside the workspace"):
        await workspace.read_file("escape.txt")

    assert await workspace.list_directory(".") == []


def test_local_workspace_rejects_missing_root(tmp_path) -> None:
    with pytest.raises(ValueError, match="Workspace root is not a directory"):
        LocalWorkspace(tmp_path / "missing")


async def _read_pid(pid_file: Path) -> int:
    for _ in range(50):
        if pid_file.is_file() and (content := pid_file.read_text().strip()):
            return int(content)
        await asyncio.sleep(0.05)
    raise AssertionError(f"child never wrote its pid to {pid_file}")


async def _process_exited(pid: int) -> bool:
    for _ in range(40):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.05)
    return False


def _kill_process_if_running(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
