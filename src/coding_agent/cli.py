"""Command-line interface for Coding Agent Harness."""

from typing import Annotated

import typer

from coding_agent import __version__
from coding_agent.events.jsonl import JsonlEventStore
from coding_agent.events.store import EventStoreError

app = typer.Typer(help="A small, extensible harness for coding agents.")


def _show_version(value: bool) -> None:
    if value:
        typer.echo(f"coding-agent {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Show the version and exit."),
    ] = False,
) -> None:
    """Run Coding Agent Harness."""


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