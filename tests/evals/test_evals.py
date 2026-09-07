import asyncio
import json
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from codekeel.agent.state import AgentState, RunStatus
from codekeel.cli import app
from codekeel.evals.dataset import DatasetTask, Task, load_dataset
from codekeel.evals.runner import run_dataset
from codekeel.evals.scorer import score
from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.store import EventStoreError, TraceReadResult
from codekeel.models.base import ModelResponse, ToolCall, Usage
from codekeel.models.fake import FakeModel
from codekeel.workspace.local import LocalWorkspace
from codekeel.workspace.models import CommandResult

FIXTURES = Path(__file__).resolve().parents[2] / "evals" / "tasks"


def final_model():
    return FakeModel([ModelResponse(content='{"success":true,"verification_result":true}',
                                   usage=Usage(input_tokens=12, output_tokens=4, cost=0.02))])


class FakeWorkspace:
    def __init__(self, result=None):
        self.result = result or CommandResult("ok", "", 0)
        self.closed = False
        self.commands = []

    async def execute(self, command, **kwargs):
        self.commands.append((command, kwargs))
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result

    async def close(self):
        self.closed = True

    async def read_file(self, path):
        raise FileNotFoundError(path)

    async def write_file(self, path, content):
        raise NotImplementedError

    async def inspect_path(self, path):
        raise NotImplementedError

    async def list_directory(self, path, *, recursive=False):
        raise NotImplementedError


async def test_three_deterministic_fixtures_and_persisted_metrics(tmp_path):
    evaluation = await run_dataset(load_dataset(FIXTURES), model_factory=final_model, root=tmp_path)
    results = {item.task_id: item for item in evaluation.results}
    assert results["pass"].success and results["pass"].verification_result is True
    assert not results["fail"].success and results["fail"].verification_result is False
    assert not results["timeout"].success
    assert results["timeout"].status in {RunStatus.TIMEOUT, RunStatus.VERIFICATION_FAILED}
    assert len({item.run_id for item in evaluation.results}) == 3
    saved = [json.loads(line) for line in (evaluation.directory / "results.jsonl").read_text().splitlines()]
    assert saved == [item.model_dump(mode="json") for item in evaluation.results]
    for item in evaluation.results:
        assert item.steps == item.model_calls == item.verification_commands == 1
        assert item.tool_calls == 0 and item.input_tokens == 12 and item.output_tokens == 4
        assert item.cost == 0.02 and item.duration > 0
        trace = JsonlEventStore(evaluation.directory).read(item.run_id)
        assert trace.warning is None and trace.events[0].type == "RunStarted"
        assert Path(item.trace_path).is_file()


async def test_scripted_repair_does_not_change_source_or_next_run(tmp_path):
    entry = load_dataset(FIXTURES / "fail.yaml")[0]
    before = (entry.repository() / "parser.py").read_bytes()
    paths = []

    def workspace_factory(path):
        paths.append(path)
        assert (path / "parser.py").read_bytes() == before
        return LocalWorkspace(path)

    def repair_model():
        return FakeModel([
            ModelResponse(tool_calls=[ToolCall(id="repair", name="write_file", arguments={
                "path": "parser.py", "content": 'def parse(text):\n    if not text:\n        raise ValueError()\n'
                                                '    return text.strip()\n',
            })]), ModelResponse(content="fixed"),
        ])

    repaired = await run_dataset([entry], model_factory=repair_model,
                                 workspace_factory=workspace_factory, root=tmp_path)
    unchanged = await run_dataset([entry], model_factory=final_model,
                                  workspace_factory=workspace_factory, root=tmp_path)
    assert repaired.results[0].success and repaired.results[0].steps == 2
    assert repaired.results[0].tool_calls == 1
    assert not unchanged.results[0].success
    assert paths[0] != paths[1] and not any(path.exists() for path in paths)
    assert repaired.directory != unchanged.directory
    assert (entry.repository() / "parser.py").read_bytes() == before


@pytest.mark.parametrize("outcome,verified", [
    (CommandResult("all passed", "", 1), False),
    (CommandResult("all passed", "", 0, timed_out=True), False),
    (TimeoutError(), None), (RuntimeError("API_KEY=secret"), None),
])
async def test_fake_failures_not_model_claims_determine_score(tmp_path, outcome, verified):
    workspace = FakeWorkspace(outcome)
    evaluation = await run_dataset(load_dataset(FIXTURES / "pass.yaml"), model_factory=final_model,
                                   workspace_factory=lambda _: workspace, root=tmp_path)
    result = evaluation.results[0]
    assert not result.success and result.verification_result is verified
    assert workspace.closed
    assert "secret" not in (evaluation.directory / "results.jsonl").read_text()


async def test_fresh_models_workspaces_state_and_continuation_after_failure(tmp_path):
    models, workspaces = [], []

    def model_factory():
        model = FakeModel([]) if not models else final_model()
        models.append(model)
        return model

    def workspace_factory(path):
        workspace = FakeWorkspace()
        workspaces.append(workspace)
        return workspace

    evaluation = await run_dataset(load_dataset(FIXTURES), model_factory=model_factory,
                                   workspace_factory=workspace_factory, root=tmp_path)
    assert len(models) == len(workspaces) == 3
    assert not evaluation.results[0].success
    assert all(item.success for item in evaluation.results[1:])
    assert all(workspace.closed for workspace in workspaces)
    assert [item.model_calls for item in evaluation.results] == [1, 1, 1]
    assert [item.input_tokens for item in evaluation.results] == [0, 12, 12]
    for workspace in workspaces[1:]:
        assert workspace.commands[0][1]["inherit_env"] is False


async def test_step_limit_prevents_further_model_calls_and_no_verification_is_not_success(tmp_path):
    entry = load_dataset(FIXTURES / "pass.yaml")[0]
    data = entry.task.model_dump()
    data["limits"]["max_steps"] = 1
    entry = DatasetTask(Task.model_validate(data), entry.source)
    model = FakeModel([ModelResponse(tool_calls=[ToolCall(id="one", name="shell",
                                                        arguments={"command": "echo okay"})])])
    evaluation = await run_dataset([entry], model_factory=lambda: model,
                                   workspace_factory=lambda _: FakeWorkspace(), root=tmp_path)
    result = evaluation.results[0]
    assert result.status is RunStatus.MAX_STEPS and result.model_calls == 1
    assert result.verification_result is None and not result.success


async def test_model_deadline_and_cancellation_cleanup(tmp_path):
    workspaces = []

    class SlowModel:
        async def complete(self, messages, *, tools=None):
            await asyncio.sleep(60)
            return ModelResponse(content="done")

    def workspace_factory(path):
        workspace = FakeWorkspace()
        workspaces.append((path, workspace))
        return workspace

    evaluation = await run_dataset(load_dataset(FIXTURES / "timeout.yaml"), model_factory=SlowModel,
                                   workspace_factory=workspace_factory, root=tmp_path)
    assert evaluation.results[0].status is RunStatus.TIMEOUT
    assert evaluation.results[0].verification_result is None
    running = asyncio.create_task(run_dataset(load_dataset(FIXTURES / "pass.yaml"), model_factory=SlowModel,
                                              workspace_factory=workspace_factory, root=tmp_path))
    await asyncio.sleep(0.05)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert len(workspaces) == 2
    assert all(workspace.closed and not path.exists() for path, workspace in workspaces)


def task_file(tmp_path, **updates):
    (tmp_path / "repo").mkdir(exist_ok=True)
    data = {"id": "example", "repo": "repo", "task": "Fix it", "verification": {"command": "echo checked"}}
    data.update(updates)
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("changes", [
    {"id": "../../escape"}, {"id": "bad\nname"}, {"repo": "../outside"}, {"repo": "/tmp"},
    {"task": " "}, {"verification": {"command": "echo\x00oops"}},
    {"verification": {"command": ""}}, {"limits": {"max_steps": True}},
    {"limits": {"timeout": float("inf")}}, {"limits": {"timeout": 0}},
    {"limits": {"max_steps": -1}}, {"unknown": True}, {"verification": {"command": "echo", "judge": "llm"}},
])
def test_invalid_tasks_rejected(tmp_path, changes):
    with pytest.raises((ValidationError, ValueError)):
        load_dataset(task_file(tmp_path, **changes))


def test_loader_order_duplicates_missing_empty_and_unsafe_yaml(tmp_path):
    with pytest.raises(ValueError, match="no YAML"):
        load_dataset(tmp_path)
    path = task_file(tmp_path)
    assert load_dataset(path)[0].repository() == tmp_path / "repo"
    (tmp_path / "aaa.yml").write_text(path.read_text().replace("example", "first"))
    assert [entry.task.id for entry in load_dataset(tmp_path)] == ["first", "example"]
    (tmp_path / "duplicate.yaml").write_text(path.read_text())
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(tmp_path)
    path.write_text("!!python/object/apply:os.system ['touch PWNED']")
    with pytest.raises(yaml.YAMLError):
        load_dataset(path)
    assert not (tmp_path / "PWNED").exists()
    path.write_text("{}")
    with pytest.raises(ValidationError):
        load_dataset(path)
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "missing.yaml")


@pytest.mark.parametrize("kind", ["repo_link", "task_link", "file_link", "directory_link", "fifo"])
async def test_links_and_special_files_rejected_without_touching_target(tmp_path, kind):
    source = tmp_path / "source"
    source.mkdir()
    path = task_file(source)
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("untouched")
    if kind == "repo_link":
        (source / "alias").symlink_to(outside, target_is_directory=True)
        path = task_file(source, repo="alias")
    elif kind == "task_link":
        alias = source / "alias.yaml"
        alias.symlink_to(path)
        path = alias
    elif kind == "file_link":
        (source / "repo" / "link").symlink_to(sentinel)
    elif kind == "directory_link":
        (source / "repo" / "link").symlink_to(outside, target_is_directory=True)
    else:
        os.mkfifo(source / "repo" / "pipe")
    if kind in {"repo_link", "task_link"}:
        with pytest.raises(ValueError):
            load_dataset(path)
    else:
        def unexpected_model():
            pytest.fail("Unsafe copies must fail before model construction")
        evaluation = await run_dataset(load_dataset(path), model_factory=unexpected_model, root=tmp_path / "output")
        assert not evaluation.results[0].success
        assert evaluation.results[0].trace_path is None and evaluation.results[0].error
    assert sentinel.read_text() == "untouched"


async def test_storage_failure_propagates_and_closes_workspace(tmp_path, monkeypatch):
    workspace = FakeWorkspace()

    def broken_append(self, event):
        raise EventStoreError("disk failed")

    monkeypatch.setattr(JsonlEventStore, "append", broken_append)
    with pytest.raises(EventStoreError):
        await run_dataset(load_dataset(FIXTURES / "pass.yaml"), model_factory=final_model,
                          workspace_factory=lambda _: workspace, root=tmp_path)
    assert workspace.closed and workspace.commands == []


def test_completed_state_without_verification_evidence_cannot_score_success():
    state = AgentState(status=RunStatus.COMPLETED, verification_passed=True)
    result = score("example", "run", state, TraceReadResult(()), duration=0, trace_path=None)
    assert not result.success and result.verification_result is None


@pytest.mark.parametrize("name,exit_code", [("pass", 0), ("fail", 1), ("timeout", 1)])
def test_cli_eval_and_inspect(tmp_path, monkeypatch, name, exit_code):
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: final_model())
    result = CliRunner().invoke(app, ["eval", str(FIXTURES / f"{name}.yaml"), "--model", "fake",
                                    "--root", str(tmp_path)])
    assert result.exit_code == exit_code, result.output
    metric, output = [json.loads(line) for line in result.output.splitlines()]
    directory = Path(output["results_path"]).parent
    inspected = CliRunner().invoke(app, ["inspect", metric["run_id"], "--root", str(directory)])
    assert inspected.exit_code == 0 and "RunStarted" in inspected.output


def test_cli_invalid_dataset_never_constructs_model(tmp_path, monkeypatch):
    path = task_file(tmp_path, repo="../escape")
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: pytest.fail("Must validate before creating model"))
    result = CliRunner().invoke(app, ["eval", str(path), "--model", "fake", "--root", str(tmp_path)])
    assert result.exit_code == 1 and "Unable to evaluate" in result.output


async def test_setup_failure_records_result_and_continues(tmp_path):
    calls = []

    def factory():
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("provider secret")
        return final_model()

    evaluation = await run_dataset(load_dataset(FIXTURES), model_factory=factory,
                                   workspace_factory=lambda _: FakeWorkspace(), root=tmp_path)
    first, *rest = evaluation.results
    assert first.error == "task_execution_failed" and first.status is RunStatus.FAILED
    assert first.trace_path is None and first.verification_result is None
    assert len({item.run_id for item in evaluation.results}) == 3
    assert all(item.success for item in rest)
    assert "provider secret" not in (evaluation.directory / "results.jsonl").read_text()


async def test_copy_excludes_runtime_data_and_never_shares_hardlinked_bytes(tmp_path):
    path = task_file(tmp_path)
    repository = tmp_path / "repo"
    source = repository / "original.txt"
    source.write_text("original")
    os.link(source, repository / "alias.txt")
    (repository / ".agent").mkdir()
    (repository / ".agent" / "private.txt").write_text("private")

    def workspace_factory(copied):
        assert not (copied / ".agent").exists()
        assert (copied / "alias.txt").stat().st_ino != source.stat().st_ino
        (copied / "alias.txt").write_text("changed")
        assert (copied / "original.txt").read_text() == "original"
        return FakeWorkspace()

    evaluation = await run_dataset(load_dataset(path), model_factory=final_model,
                                   workspace_factory=workspace_factory, root=tmp_path / "out")
    assert evaluation.results[0].success and source.read_text() == "original"


async def test_explicitly_denied_verification_never_executes(tmp_path):
    path = task_file(tmp_path, verification={"command": "git push"})
    workspace = FakeWorkspace()
    evaluation = await run_dataset(load_dataset(path), model_factory=final_model,
                                   workspace_factory=lambda _: workspace, root=tmp_path / "out")
    assert not evaluation.results[0].success and evaluation.results[0].verification_result is False
    assert workspace.commands == []


async def test_empty_and_duplicate_dataset_rejected_before_creating_outputs(tmp_path):
    entry = load_dataset(FIXTURES / "pass.yaml")[0]
    for tasks in ([], [entry, entry]):
        with pytest.raises(ValueError):
            await run_dataset(tasks, model_factory=final_model, root=tmp_path / "out")
    assert not (tmp_path / "out").exists()