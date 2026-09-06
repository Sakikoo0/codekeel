import json
import os
import subprocess

import pytest

from coding_agent.workspace import DockerWorkspace

pytestmark = pytest.mark.docker

_IMAGE = os.environ.get("CODING_AGENT_TEST_DOCKER_IMAGE", "alpine:3.20")


@pytest.fixture(scope="module", autouse=True)
def require_docker() -> None:
    try:
        subprocess.run(
            ["docker", "version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pytest.skip("Docker daemon is not available")


@pytest.fixture
async def docker_workspace(tmp_path):
    workspace = DockerWorkspace(tmp_path, image=_IMAGE)
    await workspace.start()
    try:
        yield workspace
    finally:
        await workspace.close()


async def test_docker_workspace_executes_command(docker_workspace) -> None:
    result = await docker_workspace.execute("printf 'hello from docker'")

    assert result.exit_code == 0
    assert result.stdout == "hello from docker"
    assert result.stderr == ""
    assert result.timed_out is False


async def test_docker_workspace_can_disable_environment_inheritance(docker_workspace) -> None:
    result = await docker_workspace.execute(
        "printf '%s:%s' \"$SAFE\" \"${UNSET_VALUE:-absent}\"",
        env={"SAFE": "visible"},
        inherit_env=False,
    )

    assert result.exit_code == 0
    assert result.stdout == "visible:absent"


async def test_docker_workspace_persists_files(docker_workspace, tmp_path) -> None:
    written = await docker_workspace.write_file("nested/example.txt", "from host API")
    command_result = await docker_workspace.execute("printf 'from command' > command.txt")
    read = await docker_workspace.read_file("command.txt")

    assert written.content == "from host API"
    assert (tmp_path / "nested/example.txt").read_text() == "from host API"
    assert command_result.exit_code == 0
    assert read.content == "from command"
    assert (tmp_path / "command.txt").read_text() == "from command"


async def test_docker_workspace_mount_is_contained(docker_workspace, tmp_path) -> None:
    inspect = subprocess.run(
        ["docker", "inspect", "--format", "{{json .Mounts}}", docker_workspace.container_id],
        check=True,
        capture_output=True,
        text=True,
    )
    mounts = json.loads(inspect.stdout)

    assert len(mounts) == 1
    assert mounts[0]["Source"] == str(tmp_path.resolve())
    assert mounts[0]["Destination"] == "/workspace"


async def test_docker_workspace_timeout_recovers_clean_container(docker_workspace) -> None:
    result = await docker_workspace.execute("printf before; sleep 5", timeout=0.1)
    recovered = await docker_workspace.execute("printf recovered")

    assert result.exit_code == -1
    assert result.stdout == "before"
    assert result.timed_out is True
    assert recovered.exit_code == 0
    assert recovered.stdout == "recovered"


async def test_docker_workspace_network_is_disabled_by_default(docker_workspace) -> None:
    inspect = subprocess.run(
        [
            "docker",
            "inspect",
            "--format",
            "{{.HostConfig.NetworkMode}}",
            docker_workspace.container_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert inspect.stdout.strip() == "none"


async def test_docker_workspace_close_removes_container(tmp_path) -> None:
    workspace = DockerWorkspace(tmp_path, image=_IMAGE)
    await workspace.start()
    container_id = workspace.container_id

    await workspace.close()
    inspect = subprocess.run(
        ["docker", "inspect", container_id],
        check=False,
        capture_output=True,
        text=True,
    )

    assert workspace.container_id is None
    assert inspect.returncode != 0
