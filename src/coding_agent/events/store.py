"""Append/read boundary; no filesystem dependencies in the Agent."""

import json
from dataclasses import dataclass
from threading import RLock
from typing import Protocol

from pydantic import TypeAdapter

from coding_agent.events.models import Event, RunID, UnknownEvent, parse_event

_RUN_ID_ADAPTER = TypeAdapter(RunID)


def validate_run_id(run_id: str) -> str:
    return _RUN_ID_ADAPTER.validate_python(run_id)


class EventStoreError(RuntimeError):
    """An event could not be safely appended or read."""


@dataclass(frozen=True)
class TraceReadResult:
    events: tuple[Event | UnknownEvent, ...]
    warning: str | None = None


class EventStore(Protocol):
    def append(self, event: Event) -> None: ...

    def read(self, run_id: str) -> TraceReadResult: ...


def validate_append(
    event: Event | UnknownEvent, *, run_id: str, expected_sequence: int, event_ids: set[str],
) -> None:
    if event.run_id != run_id:
        raise EventStoreError("Trace contains a different run_id")
    if event.sequence != expected_sequence:
        raise EventStoreError("Event sequence must be contiguous and start at 1")
    if event.event_id in event_ids:
        raise EventStoreError("Duplicate event_id")


class MemoryEventStore:
    """Default store; serialization isolates recorded events from caller mutation."""

    def __init__(self) -> None:
        self._runs: dict[str, list[str]] = {}
        self._ids: dict[str, set[str]] = {}
        self._lock = RLock()

    def append(self, event: Event) -> None:
        with self._lock:
            snapshot = parse_event(event.model_dump(mode="json"))
            validate_append(
                snapshot, run_id=event.run_id, expected_sequence=len(self._runs.get(event.run_id, [])) + 1,
                event_ids=self._ids.get(event.run_id, set()),
            )
            self._runs.setdefault(event.run_id, []).append(snapshot.model_dump_json())
            self._ids.setdefault(event.run_id, set()).add(snapshot.event_id)

    def read(self, run_id: str) -> TraceReadResult:
        validate_run_id(run_id)
        with self._lock:
            if run_id not in self._runs:
                raise FileNotFoundError("Run trace not found")
            return TraceReadResult(tuple(parse_event(json.loads(line)) for line in self._runs[run_id]))
