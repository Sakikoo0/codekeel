import asyncio
import json
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from codekeel.agent import Agent, RunStatus
from codekeel.cli import app
from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.store import EventStoreError, MemoryEventStore
from codekeel.models import FakeModel, ModelResponse, ToolCall, Usage
from codekeel.persistence.checkpoint import Checkpoint, WorkspaceMetadata
from codekeel.persistence.sqlite import SqliteCheckpointStore
from codekeel.persistence.store import MemoryCheckpointStore, PersistenceError, ResumeError
from codekeel.runtime.approvals import resolve_approval
from codekeel.runtime.budgets import BudgetLimits
from codekeel.runtime.policy import ActionPolicy, Risk
from codekeel.tools import ToolRegistry
from codekeel.tools.filesystem import filesystem_tools
from codekeel.tools.shell import ShellConfig, ShellTool
from codekeel.workspace import CommandResult, FileInfo, FileResult, LocalWorkspace


class Crash(BaseException):
    """Process death outside normal exception handling."""


class FakeWorkspace:
    def __init__(self):
        self.commands = []
        self.files = {}

    async def inspect_path(self, path):
        path = str(path)
        if path.startswith("/") or ".." in path.split("/"):
            raise PermissionError("Path escapes workspace")
        return FileInfo(path, path, exists=path == "." or path in self.files, is_directory=path == ".")

    async def execute(self, command, **kwargs):
        self.commands.append(command)
        return CommandResult("ok", "", 0)

    async def write_file(self, path, content):
        self.files[str(path)] = content
        return FileResult(str(path), content)


def make_agent(store, trace, workspace, responses=(), **kwargs):
    model = FakeModel(list(responses))
    model.complete = AsyncMock(wraps=model.complete)
    return Agent(model, workspace, checkpoint_store=store, event_store=trace,
                 workspace_metadata=WorkspaceMetadata(kind="custom", root="fixture"), **kwargs)


async def paused(store, trace, workspace, *, call=None, **kwargs):
    call = call or ToolCall(id="model-id", name="shell", arguments={"command": "pip install demo"})
    agent = make_agent(store, trace, workspace, [ModelResponse(tool_calls=[call], usage=Usage(input_tokens=7))],
                       **kwargs)
    assert (await agent.run("task")).status is RunStatus.WAITING_FOR_APPROVAL
    assert workspace.commands == [] and workspace.files == {}
    return agent


@pytest.mark.parametrize("approved", [True, False])
@pytest.mark.parametrize("backend", ["memory", "sqlite"])
async def test_approval_roundtrip_survives_new_stores_and_runtime(tmp_path, backend, approved):
    store = MemoryCheckpointStore() if backend == "memory" else SqliteCheckpointStore(tmp_path)
    trace = MemoryEventStore() if backend == "memory" else JsonlEventStore(tmp_path)
    workspace = FakeWorkspace()
    original = await paused(store, trace, workspace)
    snapshot = store.load(original.run_id)
    assert snapshot.ready and snapshot.pending_approval.call.arguments == {"command": "pip install demo"}
    assert trace.read(original.run_id).events[-1].type == "ApprovalRequested"
    assert not any(e.type == "RunFinished" for e in trace.read(original.run_id).events)
    if backend == "sqlite":
        store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    waiting = make_agent(store, trace, workspace, policy=ActionPolicy(mode="never"))
    assert (await waiting.resume(original.run_id)).status is RunStatus.WAITING_FOR_APPROVAL
    waiting.model.complete.assert_not_awaited()
    assert store.load(original.run_id) == snapshot
    resolve_approval(store, trace, original.run_id, snapshot.pending_approval.action_id, approved=approved)
    assert workspace.commands == []  # Recording a decision never executes it.
    fresh = make_agent(store, trace, workspace, [ModelResponse(content="done")])
    state = await fresh.resume(original.run_id)
    assert state.status is RunStatus.COMPLETED
    assert state.model_calls == state.steps == 2 and state.tool_calls == 1 and state.usage.input_tokens == 7
    assert workspace.commands == (["pip install demo"] if approved else [])
    result = json.loads(state.messages[3].content)
    assert result["is_error"] is not approved
    if not approved:
        assert "rejected by user" in result["content"]
    assert state.messages[3].tool_call_id == "model-id"
    fresh.model.complete.assert_awaited_once()
    events = trace.read(original.run_id).events
    assert [e.type for e in events].count("ToolCalled") == 1
    assert [e.type for e in events].count("ApprovalResolved") == 1
    assert store.load(original.run_id).pending_approval is None
    final = make_agent(store, trace, workspace)
    assert (await final.resume(original.run_id)).status is RunStatus.COMPLETED
    final.model.complete.assert_not_awaited()
    assert trace.read(original.run_id).events == events


@pytest.mark.parametrize("command", ["pytest", "git push", "pytest; rm -rf x"])
async def test_allow_and_deny_are_observations_without_approval(command):
    workspace, trace = FakeWorkspace(), MemoryEventStore()
    agent = Agent(FakeModel([ModelResponse(tool_calls=[ToolCall(id="c", name="shell", arguments={"command": command})]),
                             ModelResponse(content="done")]), workspace, event_store=trace)
    state = await agent.run("task")
    assert state.status is RunStatus.COMPLETED
    assert workspace.commands == ([command] if command == "pytest" else [])
    assert json.loads(state.messages[3].content)["is_error"] is (command != "pytest")
    assert not any(e.type == "ApprovalRequested" for e in trace.read(agent.run_id).events)


@pytest.mark.parametrize("wrong_id", ["stale", "../escape", "\x1b[31m", "model-id"])
async def test_wrong_action_id_cannot_claim_run(wrong_id):
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    agent = await paused(store, trace, workspace)
    before = store.load(agent.run_id)
    with pytest.raises(ResumeError):
        resolve_approval(store, trace, agent.run_id, wrong_id, approved=True)
    assert store.load(agent.run_id) == before
    assert workspace.commands == []


async def test_repeated_decisions_and_model_reused_ids_cannot_reuse_approval():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    pending = original.pending_approval
    resolve_approval(store, trace, original.run_id, pending.action_id, approved=True)
    with pytest.raises(ResumeError):
        resolve_approval(store, trace, original.run_id, pending.action_id, approved=False)
    fresh = make_agent(store, trace, workspace, [ModelResponse(tool_calls=[pending.call])])
    assert (await fresh.resume(original.run_id)).status is RunStatus.WAITING_FOR_APPROVAL
    assert fresh.pending_approval.action_id != pending.action_id
    assert workspace.commands == ["pip install demo"]
    with pytest.raises(ResumeError):
        resolve_approval(store, trace, original.run_id, pending.action_id, approved=True)


@pytest.mark.parametrize("change", ["arguments", "decision", "policy", "risk"])
async def test_checkpoint_tampering_fails_before_execution(change):
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    checkpoint = store.load(original.run_id)
    data = checkpoint.model_dump()
    data["revision"] += 1
    if change == "arguments":
        for call in (data["pending_approval"]["call"], data["state"]["messages"][-1]["tool_calls"][0]):
            call["arguments"]["command"] = "pip install changed"
    elif change == "decision":
        data["pending_approval"]["approved"] = True
    elif change == "policy":
        data["policy"]["mode"] = "never"
    else:
        data["pending_approval"]["risk"] = "low"
    store.save(Checkpoint.model_validate(data), expected_revision=checkpoint.revision)
    with pytest.raises(ResumeError):
        resolve_approval(store, trace, original.run_id, checkpoint.pending_approval.action_id, approved=True)
    fresh = make_agent(store, trace, workspace)
    with pytest.raises(ResumeError):
        await fresh.resume(original.run_id)
    fresh.model.complete.assert_not_awaited()
    assert workspace.commands == []


@pytest.mark.parametrize("change", ["no_pending", "different_call", "extra_unmatched", "not_waiting"])
async def test_only_the_exact_pending_exchange_is_a_valid_checkpoint(change):
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    data = store.load(original.run_id).model_dump()
    if change == "no_pending":
        data["pending_approval"] = None
    elif change == "different_call":
        data["state"]["messages"][-1]["tool_calls"][0]["id"] = "other"
    elif change == "extra_unmatched":
        data["state"]["messages"].insert(2, data["state"]["messages"][-1])
    else:
        data["state"]["status"] = "running"
    with pytest.raises(ValueError):
        Checkpoint.model_validate(data)


async def test_concurrent_sqlite_decisions_have_one_winner(tmp_path):
    store, trace, workspace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path), FakeWorkspace()
    original = await paused(store, trace, workspace)

    def decide(approved):
        try:
            resolve_approval(SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path), original.run_id,
                             original.pending_approval.action_id, approved=approved)
            return True
        except (ResumeError, PersistenceError):
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(decide, [True, False])) == [False, True]
    assert store.load(original.run_id).ready
    assert [e.type for e in trace.read(original.run_id).events].count("ApprovalResolved") == 1


@pytest.mark.parametrize("failure", ["request_event", "request_checkpoint", "decision_event", "decision_checkpoint"])
async def test_storage_failures_never_allow_unrecorded_execution(failure):
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    save, append = store.save, trace.append

    def fail_save(checkpoint, **kwargs):
        pending = checkpoint.pending_approval
        if checkpoint.ready and pending and ((failure == "request_checkpoint" and pending.approved is None)
                                             or (failure == "decision_checkpoint" and pending.approved is not None)):
            raise PersistenceError("disk full")
        save(checkpoint, **kwargs)

    def fail_append(event):
        if ((failure == "request_event" and event.type == "ApprovalRequested")
                or (failure == "decision_event" and event.type == "ApprovalResolved")):
            raise EventStoreError("disk full")
        append(event)

    store.save, trace.append = fail_save, fail_append
    agent = make_agent(store, trace, workspace, [ModelResponse(tool_calls=[
        ToolCall(id="c", name="shell", arguments={"command": "pip install demo"})])])
    if failure.startswith("request"):
        with pytest.raises((PersistenceError, EventStoreError)):
            await agent.run("task")
    else:
        await agent.run("task")
        with pytest.raises((PersistenceError, EventStoreError)):
            resolve_approval(store, trace, agent.run_id, agent.pending_approval.action_id, approved=True)
    assert workspace.commands == []
    assert not store.load(agent.run_id).ready
    with pytest.raises(ResumeError):
        await make_agent(store, trace, workspace).resume(agent.run_id)


async def test_crash_after_approved_side_effect_cannot_replay():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)

    async def execute(command, **kwargs):
        workspace.commands.append(command)
        raise Crash()

    workspace.execute = execute
    with pytest.raises(Crash):
        await make_agent(store, trace, workspace).resume(original.run_id)
    with pytest.raises(ResumeError):
        await make_agent(store, trace, workspace).resume(original.run_id)
    assert workspace.commands == ["pip install demo"]


async def test_approval_does_not_bypass_shell_guardrails():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    registry = ToolRegistry([ShellTool(ShellConfig(denied_commands=("pip",)))])
    original = await paused(store, trace, workspace, tool_registry=registry)
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)
    fresh = make_agent(store, trace, workspace, [ModelResponse(content="done")], tool_registry=registry)
    state = await fresh.resume(original.run_id)
    assert workspace.commands == []
    assert "denied" in json.loads(state.messages[3].content)["content"]


async def test_approval_does_not_bypass_filesystem_containment():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    registry = ToolRegistry(filesystem_tools())
    action = ToolCall(id="c", name="write_file", arguments={"path": "../secret", "content": "bad"})
    original = await paused(store, trace, workspace, call=action, tool_registry=registry,
                            policy=ActionPolicy(threshold=Risk.MEDIUM))
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)
    state = await make_agent(store, trace, workspace, [ModelResponse(content="done")], tool_registry=registry).resume(
        original.run_id
    )
    assert workspace.files == {} and json.loads(state.messages[3].content)["is_error"]


async def test_approval_does_not_reset_or_double_charge_budgets():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace, budgets=BudgetLimits(max_model_calls=1, max_tool_calls=1))
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)
    fresh = make_agent(store, trace, workspace)
    state = await fresh.resume(original.run_id)
    assert state.status is RunStatus.MAX_STEPS
    assert state.model_calls == state.tool_calls == state.steps == 1
    assert workspace.commands == ["pip install demo"]
    fresh.model.complete.assert_not_awaited()


async def test_deadline_expiring_on_resume_prevents_approved_execution():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace, clock=lambda: 0, budgets=BudgetLimits(max_wall_time=1))
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)
    times = iter([0, 0, 2, 2, 2, 2])
    state = await make_agent(store, trace, workspace, clock=lambda: next(times)).resume(original.run_id)
    assert state.status is RunStatus.TIMEOUT and workspace.commands == []


@pytest.mark.parametrize("command,approved", [("approve", True), ("reject", False)])
def test_cli_records_decision_without_constructing_model(tmp_path, monkeypatch, command, approved):
    store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    original = asyncio.run(paused(store, trace, FakeWorkspace()))
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: pytest.fail("Must not construct a model"))
    args = [command, original.run_id, original.pending_approval.action_id, "--root", str(tmp_path)]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["approved"] is approved
    assert store.load(original.run_id).pending_approval.approved is approved
    assert CliRunner().invoke(app, args).exit_code == 1


@pytest.mark.parametrize("run_id", ["missing", "../escape", "\x1b[31m"])
def test_cli_invalid_run_does_not_echo_untrusted_ids(tmp_path, run_id):
    result = CliRunner().invoke(app, ["approve", run_id, "bad", "--root", str(tmp_path)])
    assert result.exit_code == 1 and "Unable to record" in result.output
    assert "\x1b" not in result.output
    assert not (tmp_path / ".agent").exists()


def test_pending_approval_survives_three_python_processes(tmp_path):
    script = textwrap.dedent('''
        import asyncio
        import sys
        from pathlib import Path
        from codekeel.agent import Agent
        from codekeel.events.jsonl import JsonlEventStore
        from codekeel.models import FakeModel, ModelResponse, ToolCall
        from codekeel.persistence.checkpoint import WorkspaceMetadata
        from codekeel.persistence.sqlite import SqliteCheckpointStore
        from codekeel.runtime.approvals import resolve_approval
        from codekeel.runtime.policy import ActionPolicy, Risk
        from codekeel.workspace import LocalWorkspace

        root, mode = Path(sys.argv[1]), sys.argv[2]
        store, trace = SqliteCheckpointStore(root), JsonlEventStore(root)
        async def main():
            responses = [ModelResponse(tool_calls=[ToolCall(id="c", name="write_file", arguments={
                "path": "result.txt", "content": "once"})])] if mode == "start" else [ModelResponse(content="done")]
            agent = Agent(FakeModel(responses), LocalWorkspace(root), checkpoint_store=store, event_store=trace,
                          workspace_metadata=WorkspaceMetadata(kind="local", root=str(root)),
                          policy=ActionPolicy(threshold=Risk.MEDIUM))
            if mode == "start":
                assert (await agent.run("task")).status == "waiting_for_approval"
                (root / "run-id").write_text(agent.run_id)
                assert not (root / "result.txt").exists()
            else:
                run_id = (root / "run-id").read_text()
                if mode == "approve":
                    pending = store.load(run_id).pending_approval
                    resolve_approval(store, trace, run_id, pending.action_id, approved=True)
                    assert not (root / "result.txt").exists()
                else:
                    state = await agent.resume(run_id)
                    assert state.status == "completed" and state.tool_calls == 1 and state.model_calls == 2
                    assert (root / "result.txt").read_text() == "once"
        asyncio.run(main())
    ''')
    for mode in ("start", "approve", "resume"):
        result = subprocess.run([sys.executable, "-c", script, str(tmp_path), mode],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr


async def test_truthy_non_boolean_decision_is_rejected():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    with pytest.raises(ValueError):
        resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved="yes")
    with pytest.raises(ValidationError):
        type(original.pending_approval).model_validate({**original.pending_approval.model_dump(), "approved": "true"})


@pytest.mark.parametrize("path", ["../secret", ".env", "link"])
async def test_approved_local_writes_still_enforce_workspace_and_file_policy(tmp_path, path):
    repo = tmp_path / "repo"
    repo.mkdir()
    secret = tmp_path / "secret"
    secret.write_text("original")
    (repo / "link").symlink_to(secret)
    store, trace = MemoryCheckpointStore(), MemoryEventStore()
    action = ToolCall(id="c", name="write_file", arguments={"path": path, "content": "bad"})
    original = make_agent(store, trace, LocalWorkspace(repo), [ModelResponse(tool_calls=[action])],
                          policy=ActionPolicy(threshold=Risk.MEDIUM))
    assert (await original.run("task")).status is RunStatus.WAITING_FOR_APPROVAL
    resolve_approval(store, trace, original.run_id, original.pending_approval.action_id, approved=True)
    fresh = make_agent(store, trace, LocalWorkspace(repo), [ModelResponse(content="done")])
    state = await fresh.resume(original.run_id)
    assert json.loads(state.messages[3].content)["is_error"]
    assert secret.read_text() == "original" and not (repo / ".env").exists()


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_cli_decision_then_resume_completes_local_run(tmp_path, monkeypatch, decision):
    store, trace = SqliteCheckpointStore(tmp_path), JsonlEventStore(tmp_path)
    original = Agent(FakeModel([ModelResponse(tool_calls=[ToolCall(
        id="c", name="write_file", arguments={"path": "result.txt", "content": "result"})])]),
        LocalWorkspace(tmp_path), checkpoint_store=store, event_store=trace,
        workspace_metadata=WorkspaceMetadata(kind="local", root=str(tmp_path)),
        policy=ActionPolicy(threshold=Risk.MEDIUM))
    asyncio.run(original.run("task"))
    runner = CliRunner()
    answer = runner.invoke(app, [decision, original.run_id, original.pending_approval.action_id,
                                 "--root", str(tmp_path)])
    assert answer.exit_code == 0, answer.output
    monkeypatch.setattr("codekeel.cli._resume_model", lambda _: FakeModel([ModelResponse(content="done")]))
    result = runner.invoke(app, ["resume", original.run_id, "--root", str(tmp_path), "--model", "fake"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "completed"
    assert (tmp_path / "result.txt").exists() is (decision == "approve")


async def test_direct_step_cannot_advance_or_damage_a_waiting_run():
    store, trace, workspace = MemoryCheckpointStore(), MemoryEventStore(), FakeWorkspace()
    original = await paused(store, trace, workspace)
    checkpoint = store.load(original.run_id)
    with pytest.raises(RuntimeError, match="running"):
        await original.step()
    assert store.load(original.run_id) == checkpoint
    assert workspace.commands == []