import subprocess
from pathlib import Path

import pytest

from coding_agent.workspace import DockerWorkspace


class SuccessfulDockerRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        command = list(args)
        self.calls.append(command)
        stdout = "container-123\n" if command[1] == "create" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


async def test_docker_workspace_uses_secure_creation_defaults(tmp_path, monkeypatch) -> None:
    runner = SuccessfulDockerRunner()
    monkeypatch.setattr("coding_agent.workspace.docker.subprocess.run", runner)
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-anthropic-secret")
    workspace = DockerWorkspace(
        tmp_path,
        image="sandbox:test",
        cpu_limit=0.5,
        memory_limit="256m",
    )

    await workspace.start()
    create_command = runner.calls[0]

    assert create_command[:2] == ["docker", "create"]
    assert create_command[create_command.index("--network") + 1] == "none"
    assert create_command[create_command.index("--cpus") + 1] == "0.5"
    assert create_command[create_command.index("--memory") + 1] == "256m"
    assert create_command[create_command.index("--mount") + 1] == (
        f"type=bind,source={tmp_path.resolve()},target=/workspace"
    )
    rendered = " ".join(create_command)
    assert "--privileged" not in create_command
    assert "host-openai-secret" not in rendered
    assert "host-anthropic-secret" not in rendered
    assert "/.ssh" not in rendered
    assert "/.aws" not in rendered
    assert "docker.sock" not in rendered

    await workspace.close()
    assert [call[1] for call in runner.calls] == ["create", "start", "stop", "rm"]


async def test_docker_workspace_network_must_be_explicitly_enabled(tmp_path, monkeypatch) -> None:
    runner = SuccessfulDockerRunner()
    monkeypatch.setattr("coding_agent.workspace.docker.subprocess.run", runner)
    workspace = DockerWorkspace(tmp_path, image="sandbox:test", network_enabled=True)

    await workspace.start()

    assert "--network" not in runner.calls[0]


async def test_docker_workspace_can_execute_with_an_exact_environment(tmp_path, monkeypatch) -> None:
    runner = SuccessfulDockerRunner()
    monkeypatch.setattr("coding_agent.workspace.docker.subprocess.run", runner)
    workspace = DockerWorkspace(tmp_path, image="sandbox:test")

    await workspace.execute("env", env={"SAFE": "visible"}, inherit_env=False)

    exec_command = next(call for call in runner.calls if call[1] == "exec")
    container_index = exec_command.index("container-123")
    assert exec_command[container_index + 1 :] == [
        "env",
        "-i",
        "SAFE=visible",
        "sh",
        "-lc",
        "env",
    ]


async def test_docker_workspace_removes_container_when_start_fails(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        command = list(args)
        calls.append(command)
        if command[1] == "create":
            return subprocess.CompletedProcess(command, 0, stdout="container-123\n", stderr="")
        if command[1] == "start":
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="start failed")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("coding_agent.workspace.docker.subprocess.run", run)
    workspace = DockerWorkspace(tmp_path, image="sandbox:test")

    with pytest.raises(RuntimeError, match="Failed to start Docker workspace"):
        await workspace.start()

    assert [call[1] for call in calls] == ["create", "start", "rm"]
    assert workspace.container_id is None


async def test_docker_workspace_recovers_container_after_command_timeout(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def run(args, **kwargs):
        command = list(args)
        calls.append(command)
        if command[1] == "create":
            return subprocess.CompletedProcess(command, 0, stdout="container-123\n", stderr="")
        if command[1] == "exec":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial\n", stderr="warning\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr("coding_agent.workspace.docker.subprocess.run", run)
    workspace = DockerWorkspace(tmp_path, image="sandbox:test")

    result = await workspace.execute("sleep 60", timeout=0.1)

    assert result.stdout == "partial\n"
    assert result.stderr == "warning\n"
    assert result.exit_code == -1
    assert result.timed_out is True
    assert [call[1] for call in calls] == ["create", "start", "exec", "kill", "start"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"image": ""}, "image cannot be blank"),
        ({"image": "--privileged"}, "interpreted as an option"),
        ({"image": "sandbox:test", "working_dir": "/"}, "absolute non-root"),
        ({"image": "sandbox:test", "working_dir": "/workspace,readonly"}, "cannot contain commas"),
        ({"image": "sandbox:test", "cpu_limit": 0}, "greater than zero"),
        ({"image": "sandbox:test", "memory_limit": ""}, "cannot be blank"),
        ({"image": "sandbox:test", "command_timeout": 0}, "greater than zero"),
    ],
)
def test_docker_workspace_rejects_unsafe_or_invalid_config(tmp_path, kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        DockerWorkspace(tmp_path, **kwargs)


def test_docker_workspace_refuses_host_root_mount() -> None:
    with pytest.raises(ValueError, match="host filesystem root"):
        DockerWorkspace("/", image="sandbox:test")


def test_docker_workspace_refuses_mount_containing_home_credentials() -> None:
    with pytest.raises(ValueError, match="sensitive directory"):
        DockerWorkspace(Path.home(), image="sandbox:test")


@pytest.mark.parametrize("path", ["../etc/passwd", "/etc/passwd"])
def test_docker_workspace_container_path_rejects_workspace_escape(tmp_path, path) -> None:
    workspace = DockerWorkspace(tmp_path, image="sandbox:test")

    with pytest.raises(PermissionError, match="outside the workspace"):
        workspace._container_path(path)
