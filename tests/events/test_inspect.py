import json
import subprocess
import sys

from typer.testing import CliRunner

from coding_agent.cli import app
from coding_agent.events.jsonl import JsonlEventStore
from coding_agent.events.models import RunStarted
from coding_agent.models import Message
from coding_agent.runtime import BudgetLimits


def test_inspect_prints_trace_and_preserves_unknown_events(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    store = JsonlEventStore()
    event = RunStarted(run_id="demo", sequence=1, payload={
        "messages": [Message(role="user", content="task\n\x1b[31m")], "budgets": BudgetLimits(),
    })
    store.append(event)
    unknown = event.model_dump(mode="json") | {
        "type": "FutureEvent", "sequence": 2, "event_id": "future", "payload": {"next": "value"},
    }
    path = tmp_path / ".agent/runs/demo/events.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps(unknown) + "\n")
    result = CliRunner().invoke(app, ["inspect", "demo"])
    assert result.exit_code == 0
    assert [json.loads(line)["type"] for line in result.stdout.splitlines()] == ["RunStarted", "FutureEvent"]
    assert "\x1b" not in result.stdout
    assert "\\u001b" in result.stdout
    original = path.read_bytes()
    with path.open("ab") as stream:
        stream.write(b"broken")
    result = CliRunner().invoke(app, ["inspect", "demo"])
    assert result.exit_code == 0
    assert "Ignored corrupt final line 3" in result.stderr
    assert len(result.stdout.splitlines()) == 2
    assert path.read_bytes() == original + b"broken"


def test_inspect_missing_or_unsafe_id_fails_without_creating_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for run_id in ("missing", "../outside"):
        result = CliRunner().invoke(app, ["inspect", run_id])
        assert result.exit_code == 1
        assert "Unable to read trace" in result.stderr
    assert not (tmp_path / ".agent").exists()


def test_inspect_in_fresh_process_does_not_import_provider_sdk(tmp_path):
    script = '''
import importlib.abc
import sys

class BlockProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "litellm" or fullname.startswith("litellm."):
            raise AssertionError("Trace inspection must not import the provider SDK")

sys.meta_path.insert(0, BlockProvider())
from coding_agent.cli import app
from coding_agent.events.jsonl import JsonlEventStore
from coding_agent.events.models import RunStarted
from coding_agent.models import Message
from coding_agent.runtime import BudgetLimits
from typer.testing import CliRunner

JsonlEventStore().append(RunStarted(run_id="offline", sequence=1, payload={
    "messages": [Message(role="user", content="offline task")], "budgets": BudgetLimits(),
}))
result = CliRunner().invoke(app, ["inspect", "offline"])
assert result.exit_code == 0, result.output
assert "offline task" in result.stdout
assert "litellm" not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr