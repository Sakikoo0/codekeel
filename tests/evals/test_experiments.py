import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError
from typer.testing import CliRunner

from codekeel.agent import Agent
from codekeel.agent.state import RunStatus
from codekeel.cli import app
from codekeel.evals.config import ExperimentConfig, UnchangedContext, configure_agent, load_config
from codekeel.evals.dataset import DatasetTask, Task
from codekeel.evals.experiments import ComparisonReport, aggregate, run_experiments
from codekeel.evals.runner import EvaluationRun, run_dataset
from codekeel.evals.scorer import EvaluationResult
from codekeel.events.jsonl import JsonlEventStore
from codekeel.models.base import Message, ModelResponse, ToolCall, ToolResult, Usage
from codekeel.models.fake import FakeModel
from codekeel.workspace.models import CommandResult, FileInfo, FileResult


class FakeWorkspace:
    def __init__(self):
        self.closed = False
        self.commands = []
        self.files = {"README.md": "repository evidence"}

    async def execute(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult("x" * 20000 if command == "echo large" else "checked", "", 0)

    async def close(self):
        self.closed = True

    async def inspect_path(self, path):
        content = self.files.get(str(path))
        return FileInfo(str(path), str(path), content is not None, size=len(content or ""))

    async def read_file(self, path):
        return FileResult(str(path), self.files[str(path)])

    async def write_file(self, path, content):
        self.files[str(path)] = content
        return FileResult(str(path), content)

    async def list_directory(self, path, *, recursive=False):
        return [await self.inspect_path(name) for name in self.files]


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "task.yaml"
    source.write_text("unused: in-memory dataset\n")
    (tmp_path / "repo").mkdir()
    return [DatasetTask(Task(id="task", repo="repo", task="Fix the parser",
                             verification={"command": "echo verified"}), source)]


def model():
    return FakeModel([ModelResponse(content="done", usage=Usage(input_tokens=4, output_tokens=2, cost=0.1))])


@pytest.mark.parametrize("toggle", ["repo_context", "tool_output_limits", "context_compaction", "planning", "explorer"])
async def test_each_toggle_is_independently_composed(dataset, tmp_path, toggle):
    recorded, agents = [], []

    class InspectModel(FakeModel):
        async def complete(self, messages, *, tools=None):
            recorded.append((messages, tools))
            return await super().complete(messages, tools=tools)

    for enabled in (False, True):
        config = ExperimentConfig(name="test", **{toggle: enabled})
        workspace = FakeWorkspace()
        agent = Agent(InspectModel([ModelResponse(content="done")]), workspace)
        configure_agent(agent, config, model_factory=model)
        agents.append(agent)
        tool_names = {item.name for item in agent.tool_registry.definitions()}
        assert ("update_plan" in tool_names) == (toggle == "planning" and enabled)
        assert ("delegate_explore" in tool_names) == (toggle == "explorer" and enabled)
        large = ToolResult(content="x" * 20000, is_error=True)
        processed = await agent.tool_output_manager.process(large, workspace=workspace, run_id="example")
        assert processed.is_error
        assert (len(processed.content) < len(large.content)) == (toggle == "tool_output_limits" and enabled)
        messages = [Message(role="system", content="system"), Message(role="user", content="task"),
                    Message(role="assistant", content="x" * 20000)]
        prepared = agent.context_manager.prepare(messages)
        assert (len(prepared[-1].content) < 20000) == (toggle == "context_compaction" and enabled)
        assert len(messages[-1].content) == 20000
        await run_dataset(dataset, model_factory=lambda: InspectModel([ModelResponse(content="done")]),
                          workspace_factory=lambda _: FakeWorkspace(), root=tmp_path / "out", config=config)
        assert ("repository evidence" in recorded[-1][0][0].content) == (toggle == "repo_context" and enabled)
    assert agents[0].tool_registry is not agents[1].tool_registry
    assert agents[0].context_manager is not agents[1].context_manager
    assert agents[0].tool_output_manager is not agents[1].tool_output_manager


async def test_output_toggle_reaches_next_model_request(dataset, tmp_path):
    observed = []

    class InspectModel(FakeModel):
        async def complete(self, messages, *, tools=None):
            observed.append([message.model_copy(deep=True) for message in messages])
            return await super().complete(messages, tools=tools)

    for enabled in (False, True):
        def factory():
            return InspectModel([
                ModelResponse(tool_calls=[ToolCall(id="large", name="shell", arguments={"command": "echo large"})]),
                ModelResponse(content="done"),
            ])
        result = await run_dataset(dataset, model_factory=factory, workspace_factory=lambda _: FakeWorkspace(),
                                   root=tmp_path / "out",
                                   config=ExperimentConfig(name="test", tool_output_limits=enabled))
        assert result.results[0].success
        result_message = next(m for m in reversed(observed[-1]) if m.role == "tool")
        content = ToolResult.model_validate_json(result_message.content).content
        assert (len(content) <= 10000) == enabled


async def test_planning_toggle_executes_only_when_enabled(dataset, tmp_path):
    def factory():
        return FakeModel([
            ModelResponse(tool_calls=[ToolCall(id="plan", name="update_plan", arguments={"items": []})]),
            ModelResponse(content="done"),
        ])
    for enabled in (False, True):
        evaluation = await run_dataset(dataset, model_factory=factory, workspace_factory=lambda _: FakeWorkspace(),
                                       root=tmp_path / "out", config=ExperimentConfig(name="test", planning=enabled))
        result = evaluation.results[0]
        assert result.success == enabled
        trace = JsonlEventStore(evaluation.directory).read(result.run_id)
        assert any(event.type == "PlanUpdated" for event in trace.events) == enabled


async def test_explorer_separate_model_budget_and_read_only_boundary(dataset, tmp_path):
    models, workspaces = [], []

    def factory():
        responses = ([ModelResponse(tool_calls=[ToolCall(id="delegate", name="delegate_explore",
                                                         arguments={"task": "Inspect"})]),
                      ModelResponse(content="done", usage=Usage(input_tokens=3, output_tokens=2, cost=0.2))]
                     if not models else
                     [ModelResponse(tool_calls=[ToolCall(id="attack", name="write_file",
                                                        arguments={"path": "README.md", "content": "changed"})],
                                    usage=Usage(input_tokens=7, output_tokens=1, cost=0.3)),
                      ModelResponse(content="report")])
        instance = FakeModel(responses)
        models.append(instance)
        return instance

    def workspace_factory(path):
        workspace = FakeWorkspace()
        workspaces.append(workspace)
        return workspace

    evaluation = await run_dataset(dataset, model_factory=factory, workspace_factory=workspace_factory,
                                   root=tmp_path / "out", config=ExperimentConfig(name="explorer", explorer=True))
    result = evaluation.results[0]
    assert result.success and result.model_calls == 4
    assert result.input_tokens == 10 and result.output_tokens == 3 and result.cost == 0.5
    assert len(models) == 2 and workspaces[0].files["README.md"] == "repository evidence"


def metric(task_id, *, success, tokens, steps, cost, duration):
    return EvaluationResult(task_id=task_id, run_id="run-" + task_id, success=success,
                            status=RunStatus.COMPLETED if success else RunStatus.FAILED,
                            steps=steps, model_calls=steps, tool_calls=0, input_tokens=tokens, output_tokens=0,
                            cost=cost, duration=duration, verification_result=True if success else None,
                            verification_commands=1 if success else 0, trace_path=None)


def test_aggregation_includes_failed_tasks_and_exact_denominators():
    results = [metric("a", success=True, tokens=10, steps=2, cost=0.5, duration=1),
               metric("b", success=False, tokens=30, steps=6, cost=1.5, duration=3)]
    assert aggregate(results).model_dump() == {
        "tasks": 2, "successes": 1, "success_rate": 0.5, "avg_tokens": 20.0,
        "avg_steps": 4.0, "avg_cost": 1.0, "avg_duration": 2.0,
    }
    for invalid in ([], [results[0], results[0]]):
        with pytest.raises(ValueError):
            aggregate(invalid)


async def test_fake_dataset_comparison_stable_schema_and_order(dataset, tmp_path, monkeypatch):
    results = (metric("a", success=True, tokens=10, steps=2, cost=0.5, duration=1),)
    configs = [ExperimentConfig(name="baseline"), ExperimentConfig(name="planning", planning=True)]
    calls = []

    async def fake_runner(tasks, *, config, **kwargs):
        calls.append((tasks, config))
        return EvaluationRun(tmp_path, results)

    monkeypatch.setattr("codekeel.evals.experiments.run_dataset", fake_runner)
    evaluation = await run_experiments(dataset, configs, model_factory=model, root=tmp_path / "out")
    assert [config for _, config in calls] == configs
    assert all(tasks == tuple(dataset) for tasks, _ in calls)
    payload = json.loads((evaluation.directory / "results.json").read_text())
    assert set(payload) == {"schema_version", "configurations"} and payload["schema_version"] == 1
    assert [group["config"]["name"] for group in payload["configurations"]] == ["baseline", "planning"]
    assert set(payload["configurations"][0]) == {"config", "results", "summary"}
    assert payload["configurations"][0]["results"] == [results[0].model_dump(mode="json")]
    assert ComparisonReport.model_validate_json(json.dumps(payload)) == evaluation.report
    markdown = (evaluation.directory / "summary.md").read_text()
    assert markdown == evaluation.report.markdown() and "100.0%" in markdown
    assert not list(evaluation.directory.glob("*.tmp"))


async def test_config_task_isolation_failure_continuation_and_cleanup(dataset, tmp_path):
    models, workspaces, paths = [], [], []

    def factory():
        instance = FakeModel([]) if not models else model()
        models.append(instance)
        return instance

    def workspace_factory(path):
        paths.append(path)
        workspace = FakeWorkspace()
        workspaces.append(workspace)
        return workspace

    configs = [ExperimentConfig(name="first"), ExperimentConfig(name="second")]
    evaluation = await run_experiments(dataset, configs, model_factory=factory, workspace_factory=workspace_factory,
                                       root=tmp_path / "out")
    first, second = evaluation.report.configurations
    assert first.summary.success_rate == 0 and second.summary.success_rate == 1
    assert first.results[0].run_id != second.results[0].run_id
    assert len(models) == 2 and len(set(paths)) == 2
    assert all(workspace.closed for workspace in workspaces) and not any(path.exists() for path in paths)
    assert configs == [ExperimentConfig(name="first"), ExperimentConfig(name="second")]


@pytest.mark.parametrize("content", [
    "name: '../escape'", "name: 'bad|table'", 'name: "bad\\nline"', "name: a\nplanning: 'false'",
    "name: a\nexplorer: 1", "name: a\nplanning: null", "name: a\nunknown: true",
    "name: a\nname: b", "name: a\nplanning: false\nplanning: true", "[]", "{}",
    "!!python/object/apply:os.system ['touch PWNED']", "name: a\n<<: {planning: true}",
])
def test_bad_config_cannot_enable_features_or_inject_paths(tmp_path, content):
    path = tmp_path / "bad.yaml"
    path.write_text(content)
    with pytest.raises((ValueError, yaml.YAMLError)):
        load_config(path)
    assert not (tmp_path / "PWNED").exists()


def test_config_defaults_immutable_and_reject_nonregular_or_large_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("name: baseline")
    config = load_config(path)
    assert config == ExperimentConfig(name="baseline")
    with pytest.raises(ValidationError):
        config.planning = True
    alias = tmp_path / "link.yaml"
    alias.symlink_to(path)
    with pytest.raises(ValueError):
        load_config(alias)
    path.write_text("x" * 64001)
    with pytest.raises(ValueError):
        load_config(path)


async def test_invalid_comparison_rejected_before_outputs(dataset, tmp_path):
    config = ExperimentConfig(name="a")
    for tasks, configs in (([], [config]), (dataset, []), (dataset, [config, config]), (dataset * 2, [config])):
        with pytest.raises(ValueError):
            await run_experiments(tasks, configs, model_factory=model, root=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_unchanged_context_deep_copies_without_mutating_original():
    original = [Message(role="assistant", tool_calls=[ToolCall(id="a", name="shell", arguments={"x": [1]})])]
    copied = UnchangedContext().prepare(original)
    copied[0].tool_calls[0].arguments["x"].append(2)
    assert original[0].tool_calls[0].arguments == {"x": [1]}


async def test_cannot_reconfigure_started_agent():
    agent = Agent(model(), FakeWorkspace())
    await agent.run("task")
    with pytest.raises(ValueError):
        configure_agent(agent, ExperimentConfig(name="test"), model_factory=model)


def test_cli_repeated_configs_and_dataset_option(tmp_path, monkeypatch):
    fixtures = Path(__file__).resolve().parents[2] / "evals" / "tasks" / "pass.yaml"
    configs = Path(__file__).resolve().parents[2] / "configs"
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: model())
    result = CliRunner().invoke(app, ["eval", "--dataset", str(fixtures), "--config", str(configs / "baseline.yaml"),
                                    "--config", str(configs / "full.yaml"), "--model", "fake",
                                    "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "baseline" in result.output and "full" in result.output and "100.0%" in result.output
    output = json.loads(result.output.splitlines()[-1])
    assert Path(output["results_path"]).is_file() and Path(output["summary_path"]).is_file()


@pytest.mark.parametrize("args", [[], ["--dataset", "evals/tasks/", "evals/tasks/"]])
def test_cli_requires_exactly_one_dataset(args):
    result = CliRunner().invoke(app, ["eval", "--model", "fake", *args])
    assert result.exit_code == 2 and "exactly one" in result.output


async def test_child_setup_error_is_failed_task_not_idle(dataset, tmp_path):
    calls = []

    def factory():
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("secret provider configuration")
        return model()

    run = await run_experiments(dataset, [ExperimentConfig(name="child", explorer=True),
                                          ExperimentConfig(name="next")],
                                model_factory=factory, workspace_factory=lambda _: FakeWorkspace(),
                                root=tmp_path / "out")
    first = run.report.configurations[0].results[0]
    assert first.status is RunStatus.FAILED and first.error and not first.success
    assert run.report.configurations[1].summary.success_rate == 1
    assert "secret provider" not in (run.directory / "results.json").read_text()


async def test_report_storage_failure_propagates(dataset, tmp_path, monkeypatch):
    def broken_save(directory, report):
        raise OSError("disk full")

    monkeypatch.setattr("codekeel.evals.experiments._save_report", broken_save)
    with pytest.raises(OSError, match="disk full"):
        await run_experiments(dataset, [ExperimentConfig(name="test")], model_factory=model,
                              workspace_factory=lambda _: FakeWorkspace(), root=tmp_path / "out")
    assert list((tmp_path / "out").rglob("results.jsonl"))  # Existing task records survive.


def test_cli_invalid_config_before_model_and_failed_comparison_exit(tmp_path, monkeypatch):
    fixtures = Path(__file__).resolve().parents[2] / "evals" / "tasks"
    config = tmp_path / "config.yaml"
    config.write_text("name: bad\nplanning: 'false'")
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: pytest.fail("Invalid config must not create models"))
    args = ["eval", "--dataset", str(fixtures / "fail.yaml"), "--config", str(config),
            "--model", "fake", "--root", str(tmp_path / "out")]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "Unable to evaluate" in result.output
    assert not (tmp_path / "out").exists()
    config.write_text("name: baseline")
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: model())
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1 and "0/1 (0.0%)" in result.output
