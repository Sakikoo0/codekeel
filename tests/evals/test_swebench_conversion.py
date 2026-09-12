"""Offline contracts for converting selected SWE-bench records."""

import hashlib
import importlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from codekeel.cli import app
from codekeel.evals import swebench as swebench_module
from codekeel.evals.dataset import load_dataset
from codekeel.evals.patch import PatchExportError, PatchStatus
from codekeel.evals.runner import run_dataset
from codekeel.evals.swebench import (
    DATASET_ID,
    DATASET_SPLIT,
    ConversionSummary,
    PredictionExportSummary,
    SweBenchConversionError,
    SweBenchInstance,
    _load_remote_dataset,
    _materialize_repository,
    convert_swebench_records,
    convert_swebench_subset,
    export_swebench_predictions,
    verify_swebench_patch,
)
from codekeel.models.base import ModelResponse, ToolCall
from codekeel.models.fake import FakeModel
from codekeel.runtime.policy import ActionPolicy, Decision


def git(repository: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


@pytest.fixture
def source_repository(tmp_path):
    repository = tmp_path / "source"
    repository.mkdir()
    git(repository, "init", "--quiet")
    git(repository, "config", "user.email", "tests@example.invalid")
    git(repository, "config", "user.name", "CodeKeel Tests")
    (repository / "module.py").write_text("VALUE = 1\n")
    git(repository, "add", "module.py")
    git(repository, "commit", "--quiet", "-m", "fixture")
    return repository, git(repository, "rev-parse", "HEAD")


def records(commit: str):
    return [
        {
            "instance_id": "org__beta-2",
            "repo": "example/beta",
            "base_commit": commit,
            "problem_statement": "Fix beta.\n!!python/object/apply:os.system ['never']",
            "patch": "PRIVATE_GOLD_BETA",
            "test_patch": "PRIVATE_TEST_BETA",
            "eval_script": "PRIVATE_EVAL_BETA",
            "FAIL_TO_PASS": ["PRIVATE_FAIL_BETA"],
            "PASS_TO_PASS": ["PRIVATE_PASS_BETA"],
        },
        {
            "instance_id": "org__alpha-1",
            "repo": "example/alpha",
            "base_commit": commit,
            "problem_statement": "Fix alpha.",
            "patch": "PRIVATE_GOLD_ALPHA",
            "test_patch": "PRIVATE_TEST_ALPHA",
            "eval_script": "PRIVATE_EVAL_ALPHA",
            "FAIL_TO_PASS": ["PRIVATE_FAIL_ALPHA"],
            "PASS_TO_PASS": ["PRIVATE_PASS_ALPHA"],
        },
    ]


def copied_materializer(source: Path):
    def materialize(instance: SweBenchInstance, destination: Path) -> None:
        shutil.copytree(source, destination)

    return materialize


def convert(tmp_path, source_repository, **overrides):
    source, commit = source_repository
    arguments = {
        "rows": records(commit),
        "instance_ids": ["org__beta-2", "org__alpha-1"],
        "revision": "a" * 40,
        "output": tmp_path / "converted",
        "materialize_repository": copied_materializer(source),
        "codekeel_sha": "b" * 40,
        "generated_at": "2026-09-08T00:00:00+00:00",
    }
    arguments.update(overrides)
    return convert_swebench_records(**arguments), commit


def prediction_inputs(tmp_path, source_repository, patches=None):
    convert(tmp_path, source_repository)
    dataset = tmp_path / "converted"
    run = tmp_path / "eval-run"
    patch_directory = run / "patches"
    patch_directory.mkdir(parents=True)
    contents = patches or {
        "org__alpha-1": "diff --git a/a.py b/a.py\n+Unicode = '雪'\n",
        "org__beta-2": "diff --git a/b.py b/b.py\n+VALUE = 2\n",
    }
    records = []
    for instance_id in reversed(tuple(contents)):
        payload = contents[instance_id].encode("utf-8")
        patch_path = patch_directory / f"{instance_id}.diff"
        patch_path.write_bytes(payload)
        records.append({
            "task_id": instance_id,
            "run_id": f"run-{instance_id}",
            "success": False,
            "status": "verification_failed",
            "steps": 2,
            "model_calls": 2,
            "tool_calls": 1,
            "input_tokens": 3,
            "output_tokens": 4,
            "cost": 0.0,
            "duration": 1.0,
            "verification_result": False,
            "verification_commands": 1,
            "trace_path": str(run / ".agent" / "runs" / instance_id / "events.jsonl"),
            "error": None,
            "patch_status": "empty" if not payload else "exported",
            "patch_path": str(patch_path),
            "patch_sha256": hashlib.sha256(payload).hexdigest(),
            "patch_bytes": len(payload),
        })
    (run / "results.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8",
    )
    return dataset, run, records


def fixed_swebench_prediction_loader(path: Path):
    """Relevant loader contract pinned at SWE-bench 02e7a74ffd0b707aab73d203fe87bdc7c76afc8e."""
    predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for prediction in predictions:
        if not isinstance(prediction, dict) or "instance_id" not in prediction:
            raise ValueError("invalid official prediction")
    return predictions


def test_exports_stable_official_jsonl_from_two_patch_artifacts(tmp_path, source_repository):
    dataset, run, _ = prediction_inputs(tmp_path, source_repository)
    output = run / "predictions.jsonl"

    summary = export_swebench_predictions(
        dataset, run, model_name="codekeel/openai/gpt-5", output=output,
    )

    assert summary == PredictionExportSummary(
        revision="a" * 40,
        predictions=2,
        empty_patches=0,
        model_name_or_path="codekeel/openai/gpt-5",
        output=str(output),
    )
    raw = output.read_bytes()
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    loaded = fixed_swebench_prediction_loader(output)
    assert [item["instance_id"] for item in loaded] == ["org__alpha-1", "org__beta-2"]
    assert all(set(item) == {"instance_id", "model_name_or_path", "model_patch"} for item in loaded)
    assert loaded[0]["model_patch"].endswith("Unicode = '雪'\n")


def test_empty_patch_is_exported_and_model_final_answer_is_never_read(tmp_path, source_repository):
    dataset, run, _ = prediction_inputs(tmp_path, source_repository, {
        "org__alpha-1": "",
        "org__beta-2": "diff --git a/b.py b/b.py\n+VALUE = 2\n",
    })
    trace = run / ".agent" / "runs" / "org__alpha-1" / "events.jsonl"
    trace.parent.mkdir(parents=True)
    trace.write_text('MODEL_FINAL_PATCH="PRIVATE_FAKE_PATCH"')
    output = run / "predictions.jsonl"

    summary = export_swebench_predictions(
        dataset, run, model_name="codekeel/test-model", output=output,
    )

    predictions = fixed_swebench_prediction_loader(output)
    assert summary.empty_patches == 1
    assert predictions[0]["model_patch"] == ""
    assert "PRIVATE_FAKE_PATCH" not in output.read_text()


@pytest.mark.parametrize("failure", [
    "missing_metadata", "missing_results", "missing_task", "missing_patch", "hash_mismatch",
    "size_mismatch", "duplicate_result", "unknown_result", "invalid_result_schema", "patch_error",
])
def test_prediction_input_failures_do_not_publish_output(tmp_path, source_repository, failure):
    dataset, run, rows = prediction_inputs(tmp_path, source_repository)
    if failure == "missing_metadata":
        (dataset / "metadata.json").unlink()
    elif failure == "missing_results":
        (run / "results.jsonl").unlink()
    elif failure == "missing_task":
        rows.pop()
    elif failure == "missing_patch":
        Path(rows[0]["patch_path"]).unlink()
    elif failure == "hash_mismatch":
        Path(rows[0]["patch_path"]).write_text("tampered")
    elif failure == "size_mismatch":
        rows[0]["patch_bytes"] += 1
    elif failure == "duplicate_result":
        rows.append(rows[0])
    elif failure == "unknown_result":
        rows[0]["task_id"] = "org__unknown-9"
    elif failure == "invalid_result_schema":
        rows[0]["resolved"] = True
    elif failure == "patch_error":
        rows[0].update({
            "patch_status": PatchStatus.ERROR,
            "patch_path": None,
            "patch_sha256": None,
            "patch_bytes": None,
        })
    if failure in {
        "missing_task", "size_mismatch", "duplicate_result", "unknown_result", "invalid_result_schema", "patch_error",
    }:
        (run / "results.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
    output = run / "predictions.jsonl"

    with pytest.raises(SweBenchConversionError):
        export_swebench_predictions(
            dataset, run, model_name="codekeel/test-model", output=output,
        )
    assert not output.exists()


@pytest.mark.parametrize(("field", "value"), [
    ("dataset", "attacker/private"),
    ("revision", "main"),
    ("base_commit", "not-a-sha"),
    ("instance_id", "../escape"),
    ("task_file", "../escape.yaml"),
])
def test_prediction_metadata_schema_is_revalidated(tmp_path, source_repository, field, value):
    dataset, run, _ = prediction_inputs(tmp_path, source_repository)
    metadata = json.loads((dataset / "metadata.json").read_text())
    if field in {"base_commit", "instance_id", "task_file"}:
        metadata["instances"][0][field] = value
    else:
        metadata[field] = value
    (dataset / "metadata.json").write_text(json.dumps(metadata))

    with pytest.raises(SweBenchConversionError):
        export_swebench_predictions(
            dataset, run, model_name="codekeel/test-model", output=run / "predictions.jsonl",
        )


@pytest.mark.parametrize("model_name", [
    "https://api.example.invalid/model", "codekeel/endpoint", "codekeel/sk-secret", "openai/gpt-5",
])
def test_sensitive_or_unscoped_model_names_are_rejected(tmp_path, source_repository, model_name):
    dataset, run, _ = prediction_inputs(tmp_path, source_repository)
    output = run / "predictions.jsonl"

    with pytest.raises(SweBenchConversionError, match="sanitized"):
        export_swebench_predictions(dataset, run, model_name=model_name, output=output)
    assert not output.exists()


@pytest.mark.parametrize("attack", ["escape", "symlink"])
def test_patch_path_escape_and_symlink_are_rejected(tmp_path, source_repository, attack):
    dataset, run, rows = prediction_inputs(tmp_path, source_repository)
    patch_path = Path(rows[0]["patch_path"])
    if attack == "escape":
        outside = tmp_path / "outside.diff"
        outside.write_bytes(patch_path.read_bytes())
        rows[0]["patch_path"] = str(outside)
    else:
        actual = tmp_path / "actual.diff"
        patch_path.replace(actual)
        patch_path.symlink_to(actual)
    (run / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )

    with pytest.raises(SweBenchConversionError, match="Patch path"):
        export_swebench_predictions(
            dataset, run, model_name="codekeel/test-model", output=run / "predictions.jsonl",
        )


def test_prediction_output_write_failure_is_sanitized_and_atomic(tmp_path, source_repository, monkeypatch):
    dataset, run, _ = prediction_inputs(tmp_path, source_repository)
    output = run / "predictions.jsonl"

    def unwritable(*args, **kwargs):
        raise PermissionError("API_KEY=secret")

    monkeypatch.setattr(swebench_module.tempfile, "mkstemp", unwritable)
    with pytest.raises(SweBenchConversionError, match="write") as failure:
        export_swebench_predictions(
            dataset, run, model_name="codekeel/test-model", output=output,
        )
    assert "secret" not in str(failure.value)
    assert not output.exists()


def test_two_records_become_stable_loadable_yaml_without_private_fields(tmp_path, source_repository):
    summary, commit = convert(tmp_path, source_repository)
    output = tmp_path / "converted"

    assert summary.model_dump() == {
        "dataset": DATASET_ID,
        "revision": "a" * 40,
        "split": DATASET_SPLIT,
        "instances": 2,
        "output": str(output),
    }
    assert sorted(item.name for item in output.glob("*.yaml")) == ["org__alpha-1.yaml", "org__beta-2.yaml"]
    loaded = load_dataset(output)
    assert [item.task.id for item in loaded] == ["org__alpha-1", "org__beta-2"]
    assert loaded[1].task.task.endswith("!!python/object/apply:os.system ['never']")
    assert loaded[0].task.verification.command == (
        f"codekeel eval-verify-swebench-patch --base-commit {commit}"
    )
    assessment = ActionPolicy().assess(ToolCall(
        id="verification", name="shell", arguments={"command": loaded[0].task.verification.command},
    ))
    assert assessment.decision is not Decision.DENY
    assert commit in loaded[0].task.verification.command
    assert loaded[0].repository().is_dir()

    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["instances"] == [
        {
            "instance_id": "org__alpha-1",
            "repo": "example/alpha",
            "base_commit": commit,
            "task_file": "org__alpha-1.yaml",
            "repository": "repos/org__alpha-1",
        },
        {
            "instance_id": "org__beta-2",
            "repo": "example/beta",
            "base_commit": commit,
            "task_file": "org__beta-2.yaml",
            "repository": "repos/org__beta-2",
        },
    ]
    public_text = "\n".join(path.read_text() for path in [*output.glob("*.yaml"), output / "metadata.json"])
    for secret in ("PRIVATE_GOLD", "PRIVATE_TEST", "PRIVATE_EVAL", "PRIVATE_FAIL", "PRIVATE_PASS"):
        assert secret not in public_text


def test_trusted_swebench_verifier_requires_a_safe_nonempty_patch(source_repository):
    repository, base_commit = source_repository
    assert verify_swebench_patch(repository, base_commit) is False

    (repository / "module.py").write_text("VALUE = 2\n")
    assert verify_swebench_patch(repository, base_commit) is True

    (repository / "module.py").write_text("VALUE = 2   \n")
    with pytest.raises(PatchExportError):
        verify_swebench_patch(repository, base_commit)

    (repository / "module.py").write_text("VALUE = 2\n")
    (repository / ".env").write_text("API_KEY=secret")
    with pytest.raises(PatchExportError) as failure:
        verify_swebench_patch(repository, base_commit)
    assert "secret" not in str(failure.value)


async def test_converted_verification_runs_without_policy_conflict(tmp_path, source_repository):
    convert(tmp_path, source_repository)

    def repair_model():
        return FakeModel([
            ModelResponse(tool_calls=[ToolCall(
                id="repair", name="write_file", arguments={"path": "module.py", "content": "VALUE = 2\n"},
            )]),
            ModelResponse(content="done"),
        ])

    evaluation = await run_dataset(
        load_dataset(tmp_path / "converted"),
        model_factory=repair_model,
        root=tmp_path / "results",
    )

    assert len(evaluation.results) == 2
    assert all(result.success and result.patch_status is PatchStatus.EXPORTED for result in evaluation.results)


def test_existing_empty_output_is_replaced_only_after_success(tmp_path, source_repository):
    output = tmp_path / "converted"
    output.mkdir()
    convert(tmp_path, source_repository, output=output)
    assert (output / "metadata.json").is_file()


@pytest.mark.parametrize("contents", ["file", "directory"])
def test_nonempty_output_is_rejected_without_changes(tmp_path, source_repository, contents):
    output = tmp_path / "converted"
    output.mkdir()
    existing = output / "existing"
    existing.write_text("unchanged") if contents == "file" else existing.mkdir()
    with pytest.raises(SweBenchConversionError, match="empty"):
        convert(tmp_path, source_repository, output=output)
    assert existing.exists()


def test_failure_does_not_publish_partial_dataset(tmp_path, source_repository):
    source, commit = source_repository
    calls = 0

    def fail_second(instance, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("API_KEY=secret")
        shutil.copytree(source, destination)

    output = tmp_path / "converted"
    with pytest.raises(SweBenchConversionError, match="materialize") as failure:
        convert_swebench_records(
            records(commit),
            ["org__alpha-1", "org__beta-2"],
            revision="a" * 40,
            output=output,
            materialize_repository=fail_second,
            codekeel_sha="b" * 40,
        )
    assert "secret" not in str(failure.value)
    assert not output.exists()
    assert not list(tmp_path.glob(".converted-*"))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"instance_ids": []}, "At least one"),
        ({"instance_ids": ["org__alpha-1", "org__alpha-1"]}, "unique"),
        ({"instance_ids": ["org__missing-1"]}, "not found"),
        ({"revision": "latest"}, "40-character"),
        ({"codekeel_sha": "not-a-sha"}, "40-character"),
    ],
)
def test_invalid_selection_and_revisions_fail(tmp_path, source_repository, changes, message):
    with pytest.raises(SweBenchConversionError, match=message):
        convert(tmp_path, source_repository, **changes)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("instance_id", "../escape"),
        ("instance_id", "/absolute"),
        ("instance_id", "bad\nname"),
        ("instance_id", "bad\x00name"),
        ("repo", "https://github.com/example/repo"),
        ("repo", "../repo"),
        ("repo", "owner/../../repo"),
        ("repo", "owner/repo;touch-PWNED"),
        ("base_commit", "f" * 39),
        ("base_commit", "g" * 40),
        ("problem_statement", " "),
        ("problem_statement", "bad\x00task"),
    ],
)
def test_untrusted_selected_fields_are_rejected(tmp_path, source_repository, field, value):
    source, commit = source_repository
    row = records(commit)[1]
    requested = row["instance_id"]
    row[field] = value
    if field == "instance_id":
        requested = value
    with pytest.raises(SweBenchConversionError):
        convert_swebench_records(
            [row],
            [requested],
            revision="a" * 40,
            output=tmp_path / "converted",
            materialize_repository=copied_materializer(source),
            codekeel_sha="b" * 40,
        )


def test_duplicate_selected_dataset_record_is_rejected(tmp_path, source_repository):
    source, commit = source_repository
    row = records(commit)[0]
    with pytest.raises(SweBenchConversionError, match="more than once"):
        convert_swebench_records(
            [row, row],
            [row["instance_id"]],
            revision="a" * 40,
            output=tmp_path / "converted",
            materialize_repository=copied_materializer(source),
            codekeel_sha="b" * 40,
        )


def test_wrong_head_and_dirty_repository_are_rejected(tmp_path, source_repository):
    source, commit = source_repository
    output = tmp_path / "converted"
    with pytest.raises(SweBenchConversionError, match="HEAD"):
        convert_swebench_records(
            records("c" * 40),
            ["org__alpha-1"],
            revision="a" * 40,
            output=output,
            materialize_repository=copied_materializer(source),
            codekeel_sha="b" * 40,
        )

    def dirty(instance, destination):
        shutil.copytree(source, destination)
        (destination / "dirty.txt").write_text("dirty")

    with pytest.raises(SweBenchConversionError, match="clean"):
        convert_swebench_records(
            records(commit), ["org__alpha-1"], revision="a" * 40, output=output,
            materialize_repository=dirty, codekeel_sha="b" * 40,
        )



def test_clean_committed_symlink_is_rejected(tmp_path, source_repository):
    source, _ = source_repository
    (source / "link").symlink_to("module.py")
    git(source, "add", "link")
    git(source, "commit", "--quiet", "-m", "link")
    commit = git(source, "rev-parse", "HEAD")

    def linked(instance, destination):
        shutil.copytree(source, destination, symlinks=True)

    with pytest.raises(SweBenchConversionError, match="link or special"):
        convert_swebench_records(
            records(commit), ["org__alpha-1"], revision="a" * 40,
            output=tmp_path / "converted", materialize_repository=linked,
            codekeel_sha="b" * 40,
        )


def test_output_symlink_is_rejected(tmp_path, source_repository):
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "converted"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(SweBenchConversionError, match="link"):
        convert(tmp_path, source_repository, output=link)


def test_remote_loader_is_lazy_reports_missing_extra_and_pins_resolved_sha(monkeypatch):
    real_import = importlib.import_module

    def missing(name):
        if name == "datasets":
            raise ImportError("secret dependency error")
        return real_import(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(SweBenchConversionError, match="uv sync --group swebench") as failure:
        _load_remote_dataset("main")
    assert "secret" not in str(failure.value)

    calls = {}
    dataset_module = SimpleNamespace(load_dataset=lambda *args, **kwargs: calls.update(load=(args, kwargs)) or [])

    class Api:
        def dataset_info(self, **kwargs):
            calls["info"] = kwargs
            return SimpleNamespace(sha="a" * 40)

    hub_module = SimpleNamespace(HfApi=Api)
    monkeypatch.setattr(
        importlib, "import_module", lambda name: dataset_module if name == "datasets" else hub_module,
    )
    resolved, rows = _load_remote_dataset("fixed-tag")
    assert resolved == "a" * 40 and list(rows) == []
    assert calls["info"] == {"repo_id": DATASET_ID, "revision": "fixed-tag"}
    assert calls["load"] == ((DATASET_ID,), {"revision": "a" * 40, "split": DATASET_SPLIT})


def test_invalid_local_inputs_fail_before_remote_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "codekeel.evals.swebench._load_remote_dataset",
        lambda revision: pytest.fail("Remote loading must follow local validation"),
    )
    with pytest.raises(SweBenchConversionError, match="Instance ID"):
        convert_swebench_subset(["../escape"], revision="main", output=tmp_path / "converted")

    output = tmp_path / "nonempty"
    output.mkdir()
    (output / "existing").write_text("unchanged")
    with pytest.raises(SweBenchConversionError, match="empty"):
        convert_swebench_subset(["org__repo-1"], revision="main", output=output)


def test_repository_materializer_uses_argv_and_sanitized_git_environment(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setenv("GIT_SSH_COMMAND", "malicious secret")
    monkeypatch.setattr(subprocess, "run", run)
    instance = SweBenchInstance("org__repo-1", "owner/repo", "a" * 40, "Fix it")
    _materialize_repository(instance, tmp_path / "repo")

    assert len(calls) == 4
    assert all(isinstance(command, list) for command, _ in calls)
    assert all("shell" not in kwargs for _, kwargs in calls)
    assert all("GIT_SSH_COMMAND" not in kwargs["env"] for _, kwargs in calls)
    assert all(kwargs["env"]["GIT_TERMINAL_PROMPT"] == "0" for _, kwargs in calls)
    assert calls[1][0][-1] == "https://github.com/owner/repo.git"


def test_cli_success_and_sanitized_failure(tmp_path, monkeypatch):
    output = tmp_path / "converted"

    def success(instance_ids, *, revision, output):
        assert instance_ids == ["org__repo-1"] and revision == "a" * 40
        return ConversionSummary(revision=revision, instances=1, output=str(output))

    monkeypatch.setattr("codekeel.evals.swebench.convert_swebench_subset", success)
    arguments = [
        "eval-convert-swebench", "--instance-id", "org__repo-1",
        "--revision", "a" * 40, "--output", str(output),
    ]
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0 and result.stderr == ""
    assert json.loads(result.stdout) == {
        "dataset": DATASET_ID, "revision": "a" * 40, "split": DATASET_SPLIT,
        "instances": 1, "output": str(output),
    }

    def failure(*args, **kwargs):
        raise RuntimeError("API_KEY=secret\x1b[31m")

    monkeypatch.setattr("codekeel.evals.swebench.convert_swebench_subset", failure)
    failed = CliRunner().invoke(app, arguments)
    assert failed.exit_code == 1 and failed.stdout == ""
    assert "check IDs" in failed.stderr
    assert "secret" not in failed.output and "\x1b" not in failed.output


def test_cli_requires_all_inputs():
    result = CliRunner().invoke(app, ["eval-convert-swebench"])
    assert result.exit_code == 2
    assert "--instance-id" in result.output
    help_result = CliRunner().invoke(app, ["eval-convert-swebench", "--help"])
    assert help_result.exit_code == 0
    assert all(option in help_result.output for option in ("--instance-id", "--revision", "--output"))