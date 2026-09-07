"""CLI presentation and error contracts, independent of provider services."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codekeel.cli import app
from codekeel.models.fake import FakeModel

FIXTURES = Path(__file__).resolve().parents[2] / "evals" / "tasks"


@pytest.mark.parametrize("comparison", [False, True])
def test_stdout_is_json_and_tables_only_use_stderr(tmp_path, monkeypatch, comparison):
    # An exhausted fake gives a deterministic failed run without executing tools.
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([]))
    args = ["eval", str(FIXTURES / "pass.yaml"), "--model", "fake", "--root", str(tmp_path / "new")]
    if comparison:
        config = tmp_path / "config.yaml"
        config.write_text("name: baseline\n")
        args += ["--config", str(config)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 1
    records = [json.loads(line) for line in result.stdout.splitlines()]
    assert Path(records[-1]["results_path"]).is_file()
    if comparison:
        assert len(records) == 1 and set(records[0]) == {"results_path", "summary_path"}
        assert "| Config |" in result.stderr and "baseline" in result.stderr
    else:
        assert len(records) == 2 and records[0]["success"] is False
        assert result.stderr == ""


@pytest.mark.parametrize("problem", ["dataset", "config", "duplicate_names"])
def test_input_errors_are_specific_safe_and_precede_model_creation(tmp_path, monkeypatch, problem):
    def unexpected_model(_):
        pytest.fail("Invalid input must not construct a model")

    monkeypatch.setattr("codekeel.cli._resume_model", unexpected_model)
    dataset = FIXTURES / "pass.yaml"
    config = tmp_path / "config.yaml"
    config.write_text('name: baseline\nplanning: "secret-value\\u001b[31m"\n')
    args = []
    if problem == "dataset":
        dataset = tmp_path / "bad.yaml"
        dataset.write_text('secret-value: "\\u001b[31m"\n')
    else:
        args += ["--config", str(config)]
        if problem == "duplicate_names":
            config.write_text("name: baseline\n")
            args += ["--config", str(config)]
    result = CliRunner().invoke(app, ["eval", str(dataset), "--model", "fake",
                                    "--root", str(tmp_path / "out"), *args])
    assert result.exit_code == 1 and result.stdout == ""
    assert ("task files" if problem == "dataset" else "unique config names") in result.stderr
    assert "secret-value" not in result.output and "\x1b" not in result.output
    assert not (tmp_path / "out").exists()


def test_runtime_failure_does_not_expose_exception_text(tmp_path, monkeypatch):
    async def broken_runner(*args, **kwargs):
        raise OSError("secret-value\x1b[31m")

    monkeypatch.setattr("codekeel.evals.runner.run_dataset", broken_runner)
    result = CliRunner().invoke(app, ["eval", str(FIXTURES / "pass.yaml"), "--model", "fake",
                                    "--root", str(tmp_path)])
    assert result.exit_code == 1 and result.stdout == ""
    assert "output storage failure" in result.stderr
    assert "secret-value" not in result.output and "\x1b" not in result.output


def test_existing_output_file_is_cli_usage_error(tmp_path):
    root = tmp_path / "file"
    root.write_text("unchanged")
    result = CliRunner().invoke(app, ["eval", str(FIXTURES / "pass.yaml"), "--model", "fake", "--root", str(root)])
    assert result.exit_code == 2 and result.stdout == ""
    assert root.read_text() == "unchanged"


def test_help_explains_streams_exit_codes_and_default_harness():
    result = CliRunner().invoke(app, ["eval", "--help"])
    assert result.exit_code == 0
    text = " ".join(result.stdout.split())
    assert "stdout" in text and "stderr" in text and "Exit codes:" in text
    assert "default harness" in text and "--dataset" in text and "--config" in text
