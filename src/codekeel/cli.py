"""Command-line interface for CodeKeel."""

from pathlib import Path
from typing import Annotated, Literal

import typer

from codekeel import __version__
from codekeel.agent.state import AgentState
from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.store import EventStoreError

app = typer.Typer(help="CodeKeel, a small, extensible harness for coding agents.")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"codekeel {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """Run codekeel."""


@app.command("inspect")
def inspect_run(
    run_id: str,
    root: Annotated[Path, typer.Option(help="Trusted host root containing .agent checkpoints and traces.")] = Path("."),
) -> None:
    """Print a run's events in sequence from .agent/runs/RUN_ID/events.jsonl."""
    import json

    try:
        trace = JsonlEventStore(root).read(run_id)
    except (OSError, ValueError, EventStoreError):
        typer.echo("Unable to read trace: missing, invalid, unsafe path, or corrupt event stream.", err=True)
        raise typer.Exit(1) from None
    for event in trace.events:
        # Escape control characters from untrusted model/tool output for terminals.
        typer.echo(json.dumps(event.model_dump(mode="json"), ensure_ascii=True))
    if trace.warning:
        typer.echo(f"Warning: {trace.warning}", err=True)


def _resume_model(name: str):
    from codekeel.models.litellm import LiteLLMModel
    return LiteLLMModel(name)


def _workspace(kind: str, repo: Path, image: str | None):
    from codekeel.workspace.docker import DockerWorkspace
    from codekeel.workspace.local import LocalWorkspace

    if kind == "docker":
        if image is None:
            raise ValueError("Docker requires an image")
        return DockerWorkspace(repo, image=image)
    return LocalWorkspace(repo)


def _show_result(run_id: str, state: AgentState, root: Path) -> None:
    """Present existing runtime accounting; do not infer an unaudited git diff."""
    import json

    from codekeel.persistence.sqlite import SqliteCheckpointStore

    checkpoint = SqliteCheckpointStore(root).load(run_id)
    typer.echo(json.dumps({
        "run_id": run_id, "status": state.status, "steps": state.steps,
        "model_calls": state.model_calls, "tool_calls": state.tool_calls,
        "verification": state.verification_passed, "files_changed": None,
        "tokens": state.usage.input_tokens + state.usage.output_tokens,
        "cost": state.usage.cost, "duration": checkpoint.elapsed_seconds,
        "trace_path": str(root.resolve() / ".agent" / "runs" / run_id / "events.jsonl"),
        "action_id": checkpoint.pending_approval.action_id if checkpoint.pending_approval else None,
    }, ensure_ascii=True))
    if state.status != "completed":
        raise typer.Exit(1)


@app.command("run")
def run_task(
    task: Annotated[str, typer.Option(help="Task to execute; never prompts for input.")],
    model: Annotated[str, typer.Option(help="Provider/model identifier.")],
    repo: Annotated[Path, typer.Option(exists=True, file_okay=False, help="Repository workspace root.")] = Path("."),
    workspace: Annotated[Literal["local", "docker"], typer.Option(help="Workspace backend.")] = "local",
    image: Annotated[str | None, typer.Option(help="Required container image for --workspace docker.")] = None,
    root: Annotated[Path, typer.Option(exists=True, file_okay=False,
                                     help="Trusted host root for .agent checkpoints and traces.")] = Path("."),
    approval: Annotated[Literal["risky", "always", "never"], typer.Option(help="Action policy mode.")] = "risky",
    verify: Annotated[list[str] | None, typer.Option(help="Trusted verification command; repeat for a suite.")] = None,
) -> None:
    """Run with default tools, durable events and checkpoints. Exit 0 only on completion."""
    import asyncio

    from codekeel.agent import Agent
    from codekeel.persistence.checkpoint import RunMetadata, WorkspaceMetadata
    from codekeel.persistence.sqlite import SqliteCheckpointStore
    from codekeel.runtime.policy import ActionPolicy
    from codekeel.runtime.verification import VerificationPolicy

    if not task.strip() or not model.strip() or "\x00" in task or "\x00" in model:
        raise typer.BadParameter("Task and model must be nonblank and contain no NUL.")
    if (workspace == "docker") != (image is not None):
        raise typer.BadParameter("Use --image exactly when --workspace docker is selected.")

    async def execute():
        verification = VerificationPolicy(test_command=None, required_commands=tuple(verify)) if verify else None
        backend = _workspace(workspace, repo, image)
        try:
            agent = Agent(
                _resume_model(model), backend,
                event_store=JsonlEventStore(root), checkpoint_store=SqliteCheckpointStore(root),
                workspace_metadata=WorkspaceMetadata(kind=workspace, root=str(repo.resolve())),
                run_metadata=RunMetadata(model=model), policy=ActionPolicy(mode=approval),
                verification_policy=verification,
            )
            try:
                state = await agent.run(task)
            except Exception:
                if agent.run_id is not None:
                    _show_result(agent.run_id, agent.state, root)
                raise
            return agent.run_id, state
        finally:
            await backend.close()

    try:
        run_id, state = asyncio.run(execute())
        assert run_id is not None
        _show_result(run_id, state, root)
    except typer.Exit:
        raise
    except Exception:
        # Backend/provider messages can contain credentials or terminal controls.
        typer.echo("Unable to run task: runtime, configuration, workspace, or storage failure.", err=True)
        raise typer.Exit(1) from None


@app.command("resume")
def resume_run(
    run_id: str,
    root: Annotated[Path, typer.Option(help="Trusted host root containing .agent checkpoints and traces.")] = Path("."),
    model: Annotated[str | None, typer.Option(help="Model name; defaults to recorded run metadata.")] = None,
) -> None:
    """Continue a settled local-workspace run without replaying completed steps."""
    import asyncio

    from codekeel.agent import Agent
    from codekeel.persistence.sqlite import SqliteCheckpointStore
    from codekeel.persistence.store import ResumeError

    async def continuation():
        store = SqliteCheckpointStore(root)
        checkpoint = store.load(run_id)
        if not checkpoint.ready:
            raise ResumeError("Run was interrupted during a step; automatic replay is unsafe")
        if checkpoint.workspace.kind != "local" or not Path(checkpoint.workspace.root).is_absolute():
            raise ResumeError("CLI resume requires an absolute local workspace; inject other workspaces in Python")
        selected = model or checkpoint.metadata.model
        if not selected:
            raise ResumeError("Supply --model or record a model name in RunMetadata")
        workspace = _workspace("local", Path(checkpoint.workspace.root), None)
        try:
            agent = Agent(
                _resume_model(selected), workspace, event_store=JsonlEventStore(root), checkpoint_store=store,
                workspace_metadata=checkpoint.workspace,
            )
            return await agent.resume(run_id)
        finally:
            await workspace.close()

    try:
        state = asyncio.run(continuation())
        _show_result(run_id, state, root)
    except typer.Exit:
        raise
    except Exception as error:
        # Do not print model/provider exception text or terminal controls from persisted data.
        detail = str(error) if isinstance(error, ResumeError) else type(error).__name__
        typer.echo(f"Unable to resume run: {detail}", err=True)
        raise typer.Exit(1) from None


def _decide_run(run_id: str, action_id: str, root: Path, *, approved: bool) -> None:
    import json

    from codekeel.persistence.sqlite import SqliteCheckpointStore
    from codekeel.persistence.store import PersistenceError
    from codekeel.runtime.approvals import resolve_approval

    try:
        decision = resolve_approval(SqliteCheckpointStore(root), JsonlEventStore(root),
                                    run_id, action_id, approved=approved)
    except (OSError, ValueError, EventStoreError, PersistenceError):
        typer.echo("Unable to record approval: missing, stale, mismatched, or unsafe run/action.", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({"run_id": run_id, "action_id": decision.action_id, "approved": decision.approved}))


@app.command("approve")
def approve_run(
    run_id: str,
    action_id: str,
    root: Annotated[Path, typer.Option(help="Trusted host root containing .agent checkpoints and traces.")] = Path("."),
) -> None:
    """Approve the exact pending action; use resume to execute it."""
    _decide_run(run_id, action_id, root, approved=True)


@app.command("reject")
def reject_run(
    run_id: str,
    action_id: str,
    root: Annotated[Path, typer.Option(help="Trusted host root containing .agent checkpoints and traces.")] = Path("."),
) -> None:
    """Reject the exact pending action; use resume to return the refusal to the model."""
    _decide_run(run_id, action_id, root, approved=False)


@app.command("eval")
def evaluate_tasks(
    model: Annotated[str, typer.Option(help="Provider/model identifier; a fresh adapter per task.")],
    dataset: Annotated[Path | None, typer.Argument(exists=True, help="Trusted local YAML task file or directory.")]
    = None,
    dataset_option: Annotated[Path | None, typer.Option("--dataset", exists=True,
                                                      help="Use instead of the positional dataset.")] = None,
    config: Annotated[list[Path] | None, typer.Option("--config", exists=True, dir_okay=False,
                                                     help="Config; repeat for independent comparisons in order.")]
    = None,
    root: Annotated[Path, typer.Option(file_okay=False,
                                      help="Trusted output directory; created if missing. Defaults to .agent/evals.")]
    = Path(".agent/evals"),
) -> None:
    """Evaluate copied local repositories using one dataset source.

    JSON records and report paths go to stdout; comparison tables and errors go
    to stderr. Exit codes: 0 all tasks passed, 1 evaluation or input-file failure,
    2 invalid CLI arguments. Without --config, use the default harness.
    """
    import asyncio
    import json

    from codekeel.evals.dataset import load_dataset
    from codekeel.evals.runner import run_dataset

    if not model.strip() or "\x00" in model:
        raise typer.BadParameter("Model must be nonblank and contain no NUL.")
    if (dataset is None) == (dataset_option is None):
        raise typer.BadParameter("Supply exactly one positional dataset or --dataset.")
    try:
        tasks = load_dataset(dataset if dataset is not None else dataset_option)
    except Exception:
        typer.echo("Unable to evaluate dataset: task files are invalid, unreadable, or reference unsafe repositories.",
                   err=True)
        raise typer.Exit(1) from None
    configs = []
    if config:
        from codekeel.evals.config import load_config
        from codekeel.evals.experiments import run_experiments

        try:
            configs = [load_config(path) for path in config]
            if len({item.name for item in configs}) != len(configs):
                raise ValueError("Duplicate config names")
        except Exception:
            typer.echo("Unable to evaluate configs: check YAML fields, boolean toggles, and unique config names.",
                       err=True)
            raise typer.Exit(1) from None
    try:
        if configs:
            comparison = asyncio.run(run_experiments(tasks, configs,
                                                     model_factory=lambda: _resume_model(model), root=root))
        else:
            comparison = None
            evaluation = asyncio.run(run_dataset(tasks, model_factory=lambda: _resume_model(model), root=root))
    except Exception:
        typer.echo("Unable to evaluate dataset: workspace, runtime, or output storage failure.", err=True)
        raise typer.Exit(1) from None
    if comparison is not None:
        typer.echo(comparison.report.markdown(), err=True)
        typer.echo(json.dumps({"results_path": str(comparison.directory / "results.json"),
                               "summary_path": str(comparison.directory / "summary.md")}))
        if not all(result.success for group in comparison.report.configurations for result in group.results):
            raise typer.Exit(1)
        return
    for result in evaluation.results:
        typer.echo(json.dumps(result.model_dump(mode="json"), ensure_ascii=True))
    typer.echo(json.dumps({"results_path": str(evaluation.directory / "results.jsonl")}))
    if not all(result.success for result in evaluation.results):
        raise typer.Exit(1)