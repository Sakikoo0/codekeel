import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codekeel import __version__
from codekeel.cli import app
from codekeel.models import FakeModel, ModelResponse, ToolCall, Usage
from codekeel.persistence.sqlite import SqliteCheckpointStore
from codekeel.workspace import CommandResult, FileInfo

runner = CliRunner()


def test_version_option_reports_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.stdout == f"codekeel {__version__}\n"


def test_unknown_option_fails() -> None:
    result = runner.invoke(app, ["--not-a-real-option"])

    assert result.exit_code == 2
    assert "No such option: --not-a-real-option" in result.output


class FakeWorkspace:
    def __init__(self):
        self.commands = []
        self.closed = 0
        self.exit_code = 0

    async def inspect_path(self, path):
        if str(path) != ".":
            raise ValueError("outside workspace")
        return FileInfo(".", ".", exists=True, is_directory=True)

    async def execute(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult(stdout="ok", stderr="", exit_code=self.exit_code)

    async def close(self):
        self.closed += 1


@pytest.fixture
def cli(tmp_path, monkeypatch):
    workspace = FakeWorkspace()
    selected = []
    model = FakeModel([ModelResponse(content="done", usage=Usage(input_tokens=3, output_tokens=2, cost=0.1))])

    def backend(kind, repo, image):
        selected.append((kind, repo, image))
        return workspace

    monkeypatch.setattr("codekeel.cli._workspace", backend)
    monkeypatch.setattr("codekeel.cli._resume_model", lambda name: model)
    args = ["run", "--repo", str(tmp_path), "--root", str(tmp_path), "--task", "fix", "--model", "fake"]
    return args, workspace, selected


def test_run_result_and_completed_resume_inspect(cli, tmp_path):
    args, workspace, selected = cli
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert set(summary) == {"run_id", "status", "steps", "model_calls", "tool_calls", "verification",
                            "files_changed", "tokens", "cost", "duration", "trace_path", "action_id"}
    assert summary.items() >= {"status": "completed", "steps": 1, "model_calls": 1, "tool_calls": 0,
                               "tokens": 5, "cost": 0.1, "verification": None, "files_changed": None}.items()
    assert summary["duration"] >= 0
    assert Path(summary["trace_path"]).is_file()
    assert workspace.closed == 1
    assert selected == [("local", tmp_path, None)]
    checkpoint = SqliteCheckpointStore(tmp_path).load(summary["run_id"])
    assert checkpoint.metadata.model == "fake"
    assert checkpoint.state.messages[1].content == "fix"
    resumed = runner.invoke(app, ["resume", summary["run_id"], "--root", str(tmp_path)])
    assert resumed.exit_code == 0, resumed.output  # exhausted fake cannot make another model call
    assert json.loads(resumed.stdout) == summary
    inspected = runner.invoke(app, ["inspect", summary["run_id"], "--root", str(tmp_path)])
    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout.splitlines()[-1])["type"] == "RunFinished"
    assert workspace.closed == 2


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_run_approval_roundtrip(cli, tmp_path, monkeypatch, decision):
    args, workspace, _ = cli
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([
        ModelResponse(tool_calls=[ToolCall(id="call", name="shell", arguments={"command": "echo ok"})]),
    ]))
    waiting = runner.invoke(app, [*args, "--approval", "always"])
    assert waiting.exit_code == 1
    summary = json.loads(waiting.stdout)
    assert summary["status"] == "waiting_for_approval"
    assert workspace.commands == []
    identity = [summary["run_id"], summary["action_id"], "--root", str(tmp_path)]
    invalid = runner.invoke(app, [decision, summary["run_id"], "wrong", "--root", str(tmp_path)])
    assert invalid.exit_code == 1
    unresolved = runner.invoke(app, ["resume", summary["run_id"], "--root", str(tmp_path)])
    assert unresolved.exit_code == 1
    assert workspace.commands == []
    assert runner.invoke(app, [decision, *identity]).exit_code == 0
    assert runner.invoke(app, [decision, *identity]).exit_code == 1
    assert workspace.commands == []
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([ModelResponse(content="done")]))
    resumed = runner.invoke(app, ["resume", summary["run_id"], "--root", str(tmp_path)])
    assert resumed.exit_code == 0, resumed.output
    assert workspace.commands == (["echo ok"] if decision == "approve" else [])


def test_runtime_failure_is_sanitized_and_closes_workspace(cli, monkeypatch):
    args, workspace, _ = cli
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([]))
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert json.loads(result.stdout)["status"] == "failed"
    assert workspace.closed == 1
    assert "Traceback" not in result.output


@pytest.mark.parametrize("exit_code", [0, 1])
def test_verification_uses_workspace_exit_code(cli, exit_code, monkeypatch):
    args, workspace, _ = cli
    workspace.exit_code = exit_code
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([
        ModelResponse(content="done") for _ in range(3)
    ]))
    result = runner.invoke(app, [*args, "--verify", "pytest"])
    assert result.exit_code == exit_code
    assert json.loads(result.stdout)["verification"] is (exit_code == 0)
    assert json.loads(result.stdout)["status"] == ("completed" if exit_code == 0 else "verification_failed")
    assert workspace.commands == ["pytest"] * (1 if exit_code == 0 else 3)


@pytest.mark.parametrize("call", [
    ToolCall(id="deny", name="shell", arguments={"command": "git push"}),
    ToolCall(id="escape", name="read_file", arguments={"path": "../secret"}),
])
def test_cli_never_mode_cannot_bypass_boundaries(cli, monkeypatch, tmp_path, call):
    args, workspace, _ = cli
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([
        ModelResponse(tool_calls=[call]), ModelResponse(content="done"),
    ]))
    result = runner.invoke(app, [*args, "--approval", "never"])
    assert result.exit_code == 0, result.output
    checkpoint = SqliteCheckpointStore(tmp_path).load(json.loads(result.stdout)["run_id"])
    assert "error" in checkpoint.state.messages[-2].content.lower()
    assert workspace.commands == []


def test_docker_composition(cli, tmp_path):
    args, workspace, selected = cli
    result = runner.invoke(app, [*args, "--workspace", "docker", "--image", "python:3.12-slim"])
    assert result.exit_code == 0, result.output
    assert selected == [("docker", tmp_path, "python:3.12-slim")]
    assert workspace.closed == 1
    resumed = runner.invoke(app, ["resume", json.loads(result.stdout)["run_id"], "--root", str(tmp_path)])
    assert resumed.exit_code == 1
    assert "absolute local workspace" in resumed.output


@pytest.mark.parametrize("options", [
    ["--workspace", "invalid"], ["--workspace", "docker"], ["--image", "unused"],
    ["--repo", "/nonexistent-codekeel-repo"], ["--task", " "], ["--model", " "],
])
def test_invalid_run_options_do_not_construct_dependencies(cli, options):
    args, workspace, selected = cli
    result = runner.invoke(app, [*args, *options])
    assert result.exit_code == 2
    assert selected == []
    assert workspace.closed == 0


@pytest.mark.parametrize("command", ["resume", "inspect", "approve", "reject"])
@pytest.mark.parametrize("run_id", ["missing", "../escape", "bad\x1b[31m"])
def test_invalid_ids_fail_closed(tmp_path, command, run_id):
    args = [command, run_id]
    if command in {"approve", "reject"}:
        args.append("action")
    result = runner.invoke(app, [*args, "--root", str(tmp_path)])
    assert result.exit_code == 1
    assert "\x1b" not in result.output
    assert not (tmp_path / ".agent").exists()


def test_unified_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in ("run", "resume", "inspect", "approve", "reject"):
        assert name in result.stdout
        assert runner.invoke(app, [name, "--help"]).exit_code == 0
    for name in ("eval", "serve"):
        assert runner.invoke(app, [name]).exit_code == 2


def test_provider_construction_failure_closes_workspace(cli, monkeypatch):
    args, workspace, _ = cli

    def fail(_):
        raise RuntimeError("SECRET\x1b[31m")

    monkeypatch.setattr("codekeel.cli._resume_model", fail)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "SECRET" not in result.output
    assert "\x1b" not in result.output
    assert workspace.closed == 1


def test_storage_symlink_cannot_be_followed(cli, tmp_path):
    args, workspace, _ = cli
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".agent").symlink_to(outside, target_is_directory=True)
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert list(outside.iterdir()) == []
    assert workspace.closed == 1


def test_workspace_factory_preserves_backend_defaults(tmp_path):
    from codekeel.cli import _workspace
    from codekeel.workspace.docker import DockerWorkspace
    from codekeel.workspace.local import LocalWorkspace

    assert isinstance(_workspace("local", tmp_path, None), LocalWorkspace)
    docker = _workspace("docker", tmp_path, "python:3.12-slim")
    assert isinstance(docker, DockerWorkspace)
    assert docker.root == tmp_path
    assert docker.network_enabled is False
    assert docker.container_id is None
    with pytest.raises(ValueError):
        _workspace("local", tmp_path / "missing", None)