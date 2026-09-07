"""Command-line interface for CodeKeel."""

from pathlib import Path
from typing import Annotated

import typer

from codekeel import __version__
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
def inspect_run(run_id: str) -> None:
    """Print a run's events in sequence from .agent/runs/RUN_ID/events.jsonl."""
    import json

    try:
        trace = JsonlEventStore().read(run_id)
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


@app.command("resume")
def resume_run(
    run_id: str,
    root: Annotated[Path, typer.Option(help="Trusted host root containing .agent checkpoints and traces.")] = Path("."),
    model: Annotated[str | None, typer.Option(help="Model name; defaults to recorded run metadata.")] = None,
) -> None:
    """Continue a settled local-workspace run without replaying completed steps."""
    import asyncio
    import json

    from codekeel.agent import Agent
    from codekeel.persistence.sqlite import SqliteCheckpointStore
    from codekeel.persistence.store import PersistenceError, ResumeError
    from codekeel.workspace.local import LocalWorkspace

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
        workspace = LocalWorkspace(checkpoint.workspace.root)
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
    except (OSError, ValueError, EventStoreError, PersistenceError) as error:
        # Do not print model/provider exception text or terminal controls from persisted data.
        detail = str(error) if isinstance(error, ResumeError) else type(error).__name__
        typer.echo(f"Unable to resume run: {detail}", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({"run_id": run_id, "status": state.status, "model_calls": state.model_calls}))
    if state.status != "completed":
        raise typer.Exit(1)