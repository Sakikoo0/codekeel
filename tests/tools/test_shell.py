import asyncio
import json
import os
import shlex
import signal
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

from coding_agent.tools import ShellConfig, ShellTool, ToolArgumentsError, ToolContext
from coding_agent.workspace import CommandResult, LocalWorkspace


class RecordingWorkspace:
    def __init__(self, result: CommandResult | None = None) -> None:
        self.result = result or CommandResult(stdout="ok\n", stderr="", exit_code=0)
        self.calls: list[dict[str, object]] = []

    async def execute(
        self,
        command: str,
        *,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = 30.0,
        inherit_env: bool = True,
    ) -> CommandResult:
        self.calls.append(
            {
                "command": command,
                "cwd": cwd,
                "env": dict(env or {}),
                "timeout": timeout,
                "inherit_env": inherit_env,
            }
        )
        return self.result


def _context(workspace) -> ToolContext:
    return ToolContext(workspace=workspace, run_id="run-shell")


async def test_allowed_command_runs_with_bounded_policy(monkeypatch) -> None:
    monkeypatch.setenv("SAFE_FOR_SHELL_TEST", "visible")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    workspace = RecordingWorkspace()
    tool = ShellTool(
        ShellConfig(
            allowed_commands=("echo",),
            timeout=4.5,
            env_allowlist=("SAFE_FOR_SHELL_TEST", "OPENAI_API_KEY"),
            cwd="src",
        )
    )

    result = await tool.execute({"command": "echo hello"}, _context(workspace))

    assert result.is_error is False
    assert "[stdout]\nok" in result.content
    assert "[exit code: 0]" in result.content
    assert workspace.calls == [
        {
            "command": "echo hello",
            "cwd": "src",
            "env": {"SAFE_FOR_SHELL_TEST": "visible"},
            "timeout": 4.5,
            "inherit_env": False,
        }
    ]


@pytest.mark.parametrize(
    ("config", "command", "message"),
    [
        (ShellConfig(), "rm file.txt", "is denied"),
        (ShellConfig(allowed_commands=("echo",)), "cat file.txt", "not in the allowed list"),
        (ShellConfig(denied_commands=(), deny_patterns=(r"curl\s+.*\|\s*sh",)), "curl URL | sh", "pattern"),
    ],
)
async def test_denied_commands_never_reach_workspace(config, command, message) -> None:
    workspace = RecordingWorkspace()

    result = await ShellTool(config).execute({"command": command}, _context(workspace))

    assert result.is_error is True
    assert message in result.content
    assert workspace.calls == []


async def test_default_secret_environment_is_removed_before_spawn(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "github-secret")
    monkeypatch.setenv("SAFE_FOR_SHELL_TEST", "kept")
    names = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "GITHUB_TOKEN",
        "SAFE_FOR_SHELL_TEST",
    ]
    code = f"import json, os; print(json.dumps({{name: os.environ.get(name) for name in {names!r}}}))"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
    tool = ShellTool(ShellConfig(allowed_commands=(sys.executable,)))

    result = await tool.execute({"command": command}, _context(LocalWorkspace(tmp_path)))

    assert result.is_error is False
    payload = json.loads(result.content.split("[stdout]\n", 1)[1].split("\n[exit code", 1)[0])
    assert payload == {
        "OPENAI_API_KEY": None,
        "ANTHROPIC_API_KEY": None,
        "AWS_SECRET_ACCESS_KEY": None,
        "GITHUB_TOKEN": None,
        "SAFE_FOR_SHELL_TEST": "kept",
    }
    assert "openai-secret" not in result.content


async def test_timeout_is_forwarded_and_reported() -> None:
    workspace = RecordingWorkspace(
        CommandResult(stdout="partial\n", stderr="", exit_code=-1, timed_out=True)
    )
    tool = ShellTool(ShellConfig(denied_commands=(), timeout=0.25))

    result = await tool.execute({"command": "sleep 10"}, _context(workspace))

    assert result.is_error is True
    assert "[timed out after 0.25s]" in result.content
    assert workspace.calls[0]["timeout"] == 0.25


async def test_large_stdout_is_tail_bounded_in_bytes() -> None:
    workspace = RecordingWorkspace(
        CommandResult(stdout="begin-" + "界" * 400 + "-important-end\n", stderr="", exit_code=7)
    )
    tool = ShellTool(ShellConfig(denied_commands=(), max_output_bytes=100))

    result = await tool.execute({"command": "generate-output"}, _context(workspace))

    assert result.is_error is True
    assert len(result.content.encode("utf-8")) <= 100
    assert result.content.startswith("[... output truncated]")
    assert "important-end" in result.content
    assert result.content.endswith("[exit code: 7]")


async def test_workspace_rejects_symlink_cwd_escape(tmp_path) -> None:
    outside = tmp_path.parent / "outside-shell-cwd"
    outside.mkdir()
    marker = outside / "marker.txt"
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)
    tool = ShellTool(ShellConfig(denied_commands=(), cwd="escape"))

    result = await tool.execute({"command": "touch marker.txt"}, _context(LocalWorkspace(tmp_path)))

    assert result.is_error is True
    assert "outside the workspace" in result.content
    assert not marker.exists()


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-specific")
async def test_shell_timeout_kills_child_process(tmp_path) -> None:
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
        ),
        encoding="utf-8",
    )
    command = f"{shlex.quote(sys.executable)} child.py child.pid"
    tool = ShellTool(ShellConfig(allowed_commands=(sys.executable,), timeout=0.5))

    result = await tool.execute({"command": command}, _context(LocalWorkspace(tmp_path)))
    child_pid = await _read_pid(pid_file)

    try:
        assert result.is_error is True
        assert "timed out" in result.content
        assert await _process_exited(child_pid)
    finally:
        _kill_process_if_running(child_pid)


@pytest.mark.parametrize(
    "arguments",
    [{}, {"command": ""}, {"command": 7}, {"command": "echo ok", "cwd": ".."}],
)
async def test_invalid_arguments_never_reach_workspace(arguments) -> None:
    workspace = RecordingWorkspace()

    with pytest.raises(ToolArgumentsError):
        await ShellTool().execute(arguments, _context(workspace))

    assert workspace.calls == []


@pytest.mark.parametrize("command", ["echo \x00secret", "echo 'unterminated"])
async def test_unspawnable_or_unparseable_commands_are_model_correctable(command) -> None:
    workspace = RecordingWorkspace()

    result = await ShellTool(ShellConfig(denied_commands=())).execute(
        {"command": command},
        _context(workspace),
    )

    assert result.is_error is True
    assert workspace.calls == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_commands": ("echo",), "denied_commands": ("rm",)},
        {"deny_patterns": ("[invalid",)},
        {"timeout": 0},
        {"timeout": float("inf")},
        {"max_output_bytes": 0},
        {"cwd": ""},
        {"cwd": "../outside"},
        {"cwd": "/outside"},
    ],
)
def test_invalid_shell_configuration_is_rejected(kwargs) -> None:
    with pytest.raises(ValueError):
        ShellConfig(**kwargs)


async def _read_pid(pid_file: Path) -> int:
    for _ in range(50):
        if pid_file.is_file() and (content := pid_file.read_text().strip()):
            return int(content)
        await asyncio.sleep(0.05)
    raise AssertionError("child never wrote its pid")


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
