"""Sequential evaluation composition over the existing agent and event runtime."""

import os
import shutil
import stat
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from codekeel.agent import Agent
from codekeel.agent.state import AgentState, RunStatus
from codekeel.evals.dataset import DatasetTask
from codekeel.evals.scorer import EvaluationResult, score
from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.store import EventStoreError, TraceReadResult
from codekeel.models.base import Model
from codekeel.runtime.budgets import BudgetLimits
from codekeel.runtime.verification import VerificationPolicy
from codekeel.tools.registry import ToolRegistry, default_tool_registry
from codekeel.tools.shell import ShellConfig, ShellTool
from codekeel.workspace.base import Workspace
from codekeel.workspace.local import LocalWorkspace


@dataclass(frozen=True)
class EvaluationRun:
    directory: Path
    results: tuple[EvaluationResult, ...]


def _copy_repository(source: Path, destination: Path) -> None:
    """Copy bytes (never hardlink); reject links/devices before following them.

    Trusted, quiescent local fixtures only. LocalWorkspace is not an OS sandbox.
    """
    destination.mkdir()
    for entry in source.iterdir():
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("Fixture contains a link or special file")
        if entry.name in {".agent", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}:
            continue
        target = destination / entry.name
        if stat.S_ISDIR(mode):
            _copy_repository(entry, target)
        else:
            shutil.copy2(entry, target, follow_symlinks=False)


async def run_dataset(
    tasks: Sequence[DatasetTask], *, model_factory: Callable[[], Model],
    root: str | Path = ".agent/evals", workspace_factory: Callable[[Path], Workspace] = LocalWorkspace,
) -> EvaluationRun:
    """Factories must return fresh instances. Persist each result before continuing.

    Root is a trusted host output directory, outside the copied workspaces. Each
    invocation creates an exclusive directory and retains the normal .agent/runs
    event format there. Operational task failures do not stop later tasks;
    cancellation and storage failures propagate, never silently losing results.
    """
    if not tasks or len({entry.task.id for entry in tasks}) != len(tasks):
        raise ValueError("Evaluation requires nonempty tasks with unique IDs")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="eval-", dir=root))
    events = JsonlEventStore(directory)
    results: list[EvaluationResult] = []
    with (directory / "results.jsonl").open("x", encoding="utf-8") as output:
        for entry in tasks:
            started = time.monotonic()
            agent = None
            workspace = None
            error = None
            with tempfile.TemporaryDirectory(prefix="codekeel-eval-workspace-") as temporary:
                try:
                    copied = Path(temporary) / "repo"
                    _copy_repository(entry.repository(), copied)
                    workspace = workspace_factory(copied)
                    registry = default_tool_registry()
                    registry = ToolRegistry([
                        ShellTool(ShellConfig(timeout=entry.task.limits.timeout)),
                        *(registry.get(item.name) for item in registry.definitions() if item.name != "shell"),
                    ])
                    agent = Agent(
                        model_factory(), workspace, event_store=events, tool_registry=registry,
                        budgets=BudgetLimits(max_steps=entry.task.limits.max_steps,
                                             max_wall_time=entry.task.limits.timeout),
                        verification_policy=VerificationPolicy(test_command=entry.task.verification.command,
                                                               max_verification_attempts=1),
                    )
                    await agent.run(entry.task.task)
                except EventStoreError:
                    raise
                except Exception:
                    # Do not copy provider exception text (possibly secrets) into metrics.
                    error = "task_execution_failed"
                finally:
                    if workspace is not None:
                        await workspace.close()
            state = agent.state if agent is not None else AgentState(status=RunStatus.FAILED)
            run_id = agent.run_id if agent is not None else None
            trace = events.read(run_id) if run_id is not None else TraceReadResult(())
            trace_path = str(directory / ".agent" / "runs" / run_id / "events.jsonl") if run_id else None
            result = score(entry.task.id, run_id or uuid4().hex, state, trace,
                           duration=time.monotonic() - started, trace_path=trace_path, error=error)
            output.write(result.model_dump_json() + "\n")
            output.flush()
            os.fsync(output.fileno())
            results.append(result)
    return EvaluationRun(directory, tuple(results))