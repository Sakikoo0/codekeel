import asyncio
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from coding_agent.agent import Agent, AgentState, RunStatus
from coding_agent.cli import app
from coding_agent.events.jsonl import JsonlEventStore
from coding_agent.events.models import BudgetUpdated
from coding_agent.events.store import MemoryEventStore
from coding_agent.models import FakeModel, Message, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from coding_agent.persistence.checkpoint import Checkpoint, RunMetadata, WorkspaceMetadata
from coding_agent.persistence.sqlite import SqliteCheckpointStore
from coding_agent.persistence.store import MemoryCheckpointStore, PersistenceError, ResumeError
from coding_agent.runtime import BudgetLimits
from coding_agent.tools import ToolRegistry
from coding_agent.workspace import FileInfo, FileResult, LocalWorkspace


class Crash(BaseException):
    """Simulate hard process exit: ordinary error/finalization handlers do not run."""


class StopAfterTwo(Agent):
    async def step(self):
        await super().step()
        if self.state.steps == 2:
            raise Crash()


class FakeWorkspace:
    def __init__(self, files=None):
        self.files = files if files is not None else {}

    async def inspect_path(self, path):
        return FileInfo(str(path), str(path), exists=True, is_directory=str(path) == ".")

    async def write_file(self, path, content):
        self.files[str(path)] = content
        return FileResult(str(path), content)


class WriteTool:
    name = "record"

    def definition(self):
        return ToolDefinition(name=self.name, description="write output", parameters={"type": "object"})

    async def execute(self, arguments, context):
        await context.workspace.write_file(arguments["path"], arguments["content"])
        return ToolResult(content="written")


def model():
    result = FakeModel([ModelResponse(
        tool_calls=[ToolCall(id=str(i), name="record", arguments={"path": f"{i}.txt", "content": str(i)})],
        usage=Usage(input_tokens=10, output_tokens=5, cost=0.1),
    ) for i in range(2)])
    result.complete = AsyncMock(wraps=result.complete)
    return result


async def stopped(store, trace, *, workspace=None, clock=None, budgets=None):
    workspace = workspace if workspace is not None else FakeWorkspace()
    args = {"clock": clock} if clock is not None else {}
    agent = StopAfterTwo(model(), workspace, checkpoint_store=store, event_store=trace,
                         workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"),
                         run_metadata=RunMetadata(model="fake"), tool_registry=ToolRegistry([WriteTool()]),
                         plan={"current": "finish"}, budgets=budgets, **args)
    with pytest.raises(Crash):
        await agent.run("task")
    return agent


def resumed(store, trace, *, workspace=None, **kwargs):
    final = FakeModel([ModelResponse(content="done", usage=Usage(input_tokens=3, output_tokens=2, cost=0.05))])
    final.complete = AsyncMock(wraps=final.complete)
    return Agent(final, workspace or FakeWorkspace(), checkpoint_store=store, event_store=trace,
                 workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"),
                 tool_registry=ToolRegistry([WriteTool()]), **kwargs)


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_restart_continues_step_three_without_replay(tmp_path, backend):
    store = MemoryCheckpointStore() if backend == "memory" else SqliteCheckpointStore(tmp_path)
    trace = MemoryEventStore() if backend == "memory" else JsonlEventStore(tmp_path)
    workspace = FakeWorkspace()
    original = await stopped(store, trace, workspace=workspace)
    run_id = original.run_id
    snapshot = store.load(run_id)
    prior_events = trace.read(run_id).events
    assert snapshot.ready and snapshot.state.steps == 2 and snapshot.state.model_calls == 2
    if backend == "sqlite":
        store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    fresh = resumed(store, trace, workspace=FakeWorkspace(workspace.files))
    state = await fresh.resume(run_id)
    assert fresh.run_id == run_id and state.status is RunStatus.COMPLETED
    assert state.steps == state.model_calls == 3 and state.tool_calls == 2
    assert state.usage.input_tokens == 23 and state.usage.output_tokens == 12
    assert state.usage.cost == pytest.approx(0.25)
    assert fresh.plan == {"current": "finish"}
    fresh.model.complete.assert_awaited_once()
    original.model.complete.assert_awaited()
    assert fresh.workspace.files == {"0.txt": "0", "1.txt": "1"}
    events = trace.read(run_id).events
    assert events[:len(prior_events)] == prior_events
    assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    assert sum(event.type == "RunStarted" for event in events) == 1
    assert store.load(run_id).state == state


async def test_real_workspace_files_survive_new_runtime_objects(tmp_path):
    original = await stopped(SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path),
                             workspace=LocalWorkspace(tmp_path))
    fresh = resumed(SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path), workspace=LocalWorkspace(tmp_path))
    await fresh.resume(original.run_id)
    assert (tmp_path / "0.txt").read_text() == "0"
    assert (tmp_path / "1.txt").read_text() == "1"


async def test_summary_messages_and_opaque_plan_roundtrip(tmp_path):
    store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    original = await stopped(store, trace)
    summary = Message(role="assistant", content="Summary of previous history:\n## Goal\nfinish task")
    original.state.messages.insert(2, summary)
    original.save_checkpoint()
    fresh = resumed(SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path))
    await fresh.resume(original.run_id)
    assert fresh.state.messages[2] == summary and fresh.plan == original.plan


async def test_persisted_limits_and_elapsed_time_are_not_reset():
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    now = [10.0]
    original = await stopped(store, trace, clock=lambda: now[0], budgets=BudgetLimits(max_wall_time=20.0))
    now[0] = 29.0
    original.save_checkpoint()
    fresh = resumed(store, trace, clock=lambda: 100.0, budgets=BudgetLimits(max_wall_time=1000.0))
    await fresh.resume(original.run_id)
    assert fresh.termination_policy.budgets.max_wall_time == 20.0
    assert fresh._started_at == 81.0


@pytest.mark.parametrize("limits,status", [
    (BudgetLimits(max_model_calls=2), RunStatus.MAX_STEPS),
    (BudgetLimits(max_input_tokens=20), RunStatus.MAX_TOKENS),
    (BudgetLimits(max_cost=0.2), RunStatus.MAX_COST),
])
async def test_resume_cannot_reset_spent_budgets(limits, status):
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace, budgets=limits)
    fresh = resumed(store, trace, budgets=BudgetLimits())
    assert (await fresh.resume(original.run_id)).status is status
    fresh.model.complete.assert_not_awaited()


async def test_completed_resume_is_idempotent():
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace)
    first = resumed(store, trace)
    await first.resume(original.run_id)
    events = trace.read(original.run_id).events
    second = resumed(store, trace)
    assert (await second.resume(original.run_id)).status is RunStatus.COMPLETED
    second.model.complete.assert_not_awaited()
    assert trace.read(original.run_id).events == events


async def test_crash_after_terminal_checkpoint_finishes_trace_without_model_replay():
    class StopFinal(Agent):
        async def step(self):
            await super().step()
            raise Crash()

    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = StopFinal(FakeModel([ModelResponse(content="done")]), FakeWorkspace(),
                         checkpoint_store=store, event_store=trace,
                         workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"),
                         tool_registry=ToolRegistry([WriteTool()]))
    with pytest.raises(Crash):
        await original.run("task")
    fresh = resumed(store, trace)
    assert (await fresh.resume(original.run_id)).status is RunStatus.COMPLETED
    fresh.model.complete.assert_not_awaited()
    assert trace.read(original.run_id).events[-1].type == "RunFinished"


async def test_inflight_crash_never_replays_prior_model_or_tool():
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace)
    fresh = resumed(store, trace)
    fresh.model.complete.side_effect = Crash()
    with pytest.raises(Crash):
        await fresh.resume(original.run_id)
    assert not store.load(original.run_id).ready
    third = resumed(store, trace)
    with pytest.raises(ResumeError, match="interrupted"):
        await third.resume(original.run_id)
    third.model.complete.assert_not_awaited()


async def test_tool_side_effect_before_crash_is_never_repeated():
    class CrashingTool(WriteTool):
        async def execute(self, arguments, context):
            await super().execute(arguments, context)
            raise Crash()

    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    workspace = FakeWorkspace()
    agent = Agent(model(), workspace, checkpoint_store=store, event_store=trace,
                  workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"),
                  tool_registry=ToolRegistry([CrashingTool()]))
    with pytest.raises(Crash):
        await agent.run("task")
    assert workspace.files == {"0.txt": "0"}
    with pytest.raises(ResumeError):
        await resumed(store, trace).resume(agent.run_id)


@pytest.mark.parametrize("mismatch", ["workspace", "tools", "trace", "budget", "usage"])
async def test_mismatched_restore_fails_before_model(mismatch):
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace)
    fresh = resumed(store, trace)
    if mismatch == "workspace":
        fresh.workspace_metadata = WorkspaceMetadata(kind="custom", root="elsewhere")
    elif mismatch == "tools":
        fresh.tool_registry = ToolRegistry([])
    elif mismatch == "trace":
        previous = trace.read(original.run_id).events[-1]
        trace.append(BudgetUpdated(run_id=original.run_id, sequence=previous.sequence + 1,
                                   payload=original._budget_payload()))
    else:
        checkpoint = store.load(original.run_id)
        payload = checkpoint.model_dump(mode="json")
        payload["revision"] += 1
        if mismatch == "budget":
            payload["budgets"]["max_steps"] = 9999
        else:
            payload["state"]["usage"]["cost"] = 0
        store.save(Checkpoint.model_validate(payload), expected_revision=checkpoint.revision)
    with pytest.raises(ResumeError):
        await fresh.resume(original.run_id)
    fresh.model.complete.assert_not_awaited()


async def test_damaged_trace_tail_refuses_resume(tmp_path):
    store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    original = await stopped(store, trace)
    with (tmp_path / ".agent/runs" / original.run_id / "events.jsonl").open("ab") as stream:
        stream.write(b'{"broken"')
    with pytest.raises(ResumeError):
        await resumed(store, trace).resume(original.run_id)


async def test_checkpoint_write_failure_prevents_next_model_call():
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace)
    store.save = lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk full"))
    fresh = resumed(store, trace)
    with pytest.raises(PersistenceError):
        await fresh.resume(original.run_id)
    fresh.model.complete.assert_not_awaited()


async def test_cancelled_inflight_run_is_saved_but_not_resumable():
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    original = await stopped(store, trace)
    fresh = resumed(store, trace)
    fresh.model.complete.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await fresh.resume(original.run_id)
    checkpoint = store.load(original.run_id)
    assert not checkpoint.ready and checkpoint.state.status is RunStatus.CANCELLED


def sample():
    return Checkpoint(run_id="test", revision=1, ready=True, event_sequence=1, last_event_id="event",
                      state=AgentState(messages=[Message(role="user", content="task")]), budgets=BudgetLimits(),
                      elapsed_seconds=3.0, workspace=WorkspaceMetadata(kind="custom", root="fixture"), tools=[])


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_store_roundtrip_and_revision_conflict(tmp_path, backend):
    store = MemoryCheckpointStore() if backend == "memory" else SqliteCheckpointStore(tmp_path)
    snapshot = sample()
    store.save(snapshot, expected_revision=0)
    loaded = store.load("test")
    assert loaded == snapshot
    loaded.state.messages.clear()
    assert store.load("test").state.messages == snapshot.state.messages
    with pytest.raises(PersistenceError):
        store.save(snapshot, expected_revision=0)
    if backend == "sqlite":
        assert (tmp_path / ".agent/checkpoints.sqlite3").stat().st_mode & 0o777 == 0o600


def test_concurrent_sqlite_claims_have_only_one_winner(tmp_path):
    SqliteCheckpointStore(tmp_path).save(sample(), expected_revision=0)
    claimed = sample().model_copy(update={"revision": 2, "ready": False})

    def claim():
        try:
            SqliteCheckpointStore(tmp_path).save(claimed, expected_revision=1)
            return True
        except PersistenceError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: claim(), range(2))) == [False, True]


@pytest.mark.parametrize("run_id", ["../escape", "/tmp/x", "a/b", "x\n", "' OR 1=1 --"])
def test_invalid_run_ids_do_not_access_storage(tmp_path, run_id):
    with pytest.raises(ValueError):
        SqliteCheckpointStore(tmp_path).load(run_id)
    assert not (tmp_path / ".agent").exists()


@pytest.mark.parametrize("kind", ["directory_symlink", "db_symlink", "db_hardlink", "journal_symlink", "fifo"])
def test_unsafe_database_paths_are_rejected(tmp_path, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    secret = outside / "secret"
    secret.write_text("SECRET")
    directory = tmp_path / ".agent"
    if kind == "directory_symlink":
        directory.symlink_to(outside, target_is_directory=True)
    else:
        directory.mkdir()
        db = directory / "checkpoints.sqlite3"
        if kind == "db_symlink":
            db.symlink_to(secret)
        elif kind == "db_hardlink":
            os.link(secret, db)
        elif kind == "fifo":
            os.mkfifo(db)
        else:
            (directory / "checkpoints.sqlite3-journal").symlink_to(secret)
    with pytest.raises((OSError, PersistenceError)):
        SqliteCheckpointStore(tmp_path).save(sample(), expected_revision=0)
    assert secret.read_text() == "SECRET"


@pytest.mark.parametrize("corruption", ["invalid_json", "wrong_id", "version", "orphan"])
def test_corrupt_checkpoint_is_rejected(tmp_path, corruption):
    store = SqliteCheckpointStore(tmp_path)
    store.save(sample(), expected_revision=0)
    data = sample().model_dump(mode="json")
    if corruption == "wrong_id":
        data["run_id"] = "other"
    elif corruption == "version":
        data["schema_version"] = 2
    elif corruption == "orphan":
        data["state"]["messages"] = [{"role": "tool", "tool_call_id": "unknown"}]
    payload = "{" if corruption == "invalid_json" else json.dumps(data)
    with sqlite3.connect(tmp_path / ".agent/checkpoints.sqlite3") as connection:
        connection.execute("UPDATE checkpoints SET payload = ?", (payload,))
    with pytest.raises(PersistenceError):
        store.load("test")


def test_missing_checkpoint_read_does_not_create_database(tmp_path):
    with pytest.raises(FileNotFoundError):
        SqliteCheckpointStore(tmp_path).load("missing")
    assert not (tmp_path / ".agent").exists()


def test_cli_resume_with_new_local_workspace(tmp_path, monkeypatch):
    # Default tool schema matches the CLI. The first two steps need no real model or shell.
    class StopLocal(Agent):
        async def step(self):
            await super().step()
            if self.state.steps == 2:
                raise Crash()

    store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    source = FakeModel([ModelResponse(tool_calls=[ToolCall(id=str(i), name="write_file",
                        arguments={"path": f"{i}.txt", "content": str(i)})]) for i in range(2)])
    agent = StopLocal(source, LocalWorkspace(tmp_path), checkpoint_store=store, event_store=trace,
                      workspace_metadata=WorkspaceMetadata(kind="local", root=str(tmp_path)),
                      run_metadata=RunMetadata(model="fake"))
    with pytest.raises(Crash):
        asyncio.run(agent.run("task"))
    final = FakeModel([ModelResponse(content="done")])
    monkeypatch.setattr("coding_agent.cli._resume_model", lambda _: final)
    result = CliRunner().invoke(app, ["resume", agent.run_id, "--root", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "completed"
    assert store.load(agent.run_id).state.model_calls == 3
    assert (tmp_path / "0.txt").read_text() == "0"


def test_cli_resume_missing_run_fails_without_model(tmp_path, monkeypatch):
    def unexpected(_):
        pytest.fail("Model must not be constructed")
    monkeypatch.setattr("coding_agent.cli._resume_model", unexpected)
    result = CliRunner().invoke(app, ["resume", "missing", "--root", str(tmp_path)])
    assert result.exit_code == 1 and "Unable to resume" in result.output


def test_persistence_requires_explicit_event_store_and_workspace_metadata():
    with pytest.raises(ValueError, match="workspace metadata"):
        Agent(FakeModel([]), FakeWorkspace(), checkpoint_store=MemoryCheckpointStore())
    with pytest.raises(ValueError, match="event store"):
        Agent(FakeModel([]), FakeWorkspace(), checkpoint_store=MemoryCheckpointStore(),
              workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"))


def test_hard_process_exit_and_fresh_python_resume(tmp_path):
    script = textwrap.dedent('''
        import asyncio
        import os
        import sys
        from pathlib import Path
        from coding_agent.agent import Agent
        from coding_agent.events.jsonl import JsonlEventStore
        from coding_agent.models import FakeModel, ModelResponse, ToolCall, Usage
        from coding_agent.persistence.checkpoint import WorkspaceMetadata
        from coding_agent.persistence.sqlite import SqliteCheckpointStore
        from coding_agent.workspace import LocalWorkspace

        root = Path(sys.argv[1])
        class ExitingAgent(Agent):
            async def step(self):
                await super().step()
                if self.state.steps == 2:
                    (root / "run-id").write_text(self.run_id)
                    os._exit(0)

        async def main():
            start = sys.argv[2] == "start"
            responses = [ModelResponse(tool_calls=[ToolCall(id=str(i), name="write_file",
                         arguments={"path": f"{i}.txt", "content": str(i)})],
                         usage=Usage(input_tokens=10)) for i in range(2)] if start else [ModelResponse(content="done")]
            cls = ExitingAgent if start else Agent
            agent = cls(FakeModel(responses), LocalWorkspace(root),
                        checkpoint_store=SqliteCheckpointStore(root), event_store=JsonlEventStore(root),
                        workspace_metadata=WorkspaceMetadata(kind="local", root=str(root)))
            if start:
                await agent.run("task")
            else:
                state = await agent.resume((root / "run-id").read_text())
                assert state.status == "completed" and state.model_calls == 3
                assert state.tool_calls == 2 and state.usage.input_tokens == 20
                assert (root / "0.txt").read_text() == "0"
                assert (root / "1.txt").read_text() == "1"
        asyncio.run(main())
    ''')
    for mode in ("start", "resume"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr