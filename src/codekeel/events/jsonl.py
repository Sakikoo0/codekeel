"""Local append-only JSONL traces. POSIX, matching the local workspace backend."""

import fcntl
import json
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

from pydantic import ValidationError

from codekeel.events.models import Event, UnknownEvent, parse_event
from codekeel.events.store import EventStoreError, TraceReadResult, validate_append, validate_run_id


def _read_lines(stream: BinaryIO, run_id: str) -> TraceReadResult:
    events: list[Event | UnknownEvent] = []
    event_ids: set[str] = set()
    line = stream.readline()
    number = 0
    while line:
        number += 1
        following = stream.readline()
        try:
            event = parse_event(json.loads(line))
        except (ValueError, UnicodeError, ValidationError):
            if not following:
                return TraceReadResult(tuple(events), f"Ignored corrupt final line {number}")
            raise EventStoreError(f"Corrupt event at line {number}") from None
        validate_append(event, run_id=run_id, expected_sequence=len(events) + 1, event_ids=event_ids)
        events.append(event)
        event_ids.add(event.event_id)
        if not following and not line.endswith(b"\n"):
            return TraceReadResult(tuple(events), f"Final line {number} has no newline; trace may be incomplete")
        line = following
    return TraceReadResult(tuple(events))


class JsonlEventStore:
    """Store .agent/runs/<run_id>/events.jsonl below a trusted host directory.

    Readers tolerate only a corrupt final record; writers refuse damaged tails.
    No repair, truncation, checkpoint, or agent resume is performed.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    @contextmanager
    def _open(self, run_id: str, *, write: bool) -> Iterator[BinaryIO]:
        validate_run_id(run_id)
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for name in (".agent", "runs", run_id):
                if write:
                    try:
                        os.mkdir(name, mode=0o700, dir_fd=directory)
                    except FileExistsError:
                        pass
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                os.close(directory)
                directory = child
            flags = os.O_RDWR | os.O_CREAT | os.O_APPEND if write else os.O_RDONLY
            descriptor = os.open("events.jsonl", flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise EventStoreError("Trace must be a regular file with one link")
                with os.fdopen(descriptor, "r+b" if write else "rb", closefd=False) as stream:
                    fcntl.flock(descriptor, fcntl.LOCK_EX if write else fcntl.LOCK_SH)
                    try:
                        yield stream
                    finally:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        finally:
            os.close(directory)

    def append(self, event: Event) -> None:
        # Validate and serialize before opening, never interpolate payloads into paths.
        snapshot = parse_event(event.model_dump(mode="json"))
        encoded = (snapshot.model_dump_json() + "\n").encode("utf-8")
        with self._open(snapshot.run_id, write=True) as stream:
            current = _read_lines(stream, snapshot.run_id)
            if current.warning:
                raise EventStoreError("Refusing to append to an incomplete or corrupt trace")
            validate_append(
                snapshot, run_id=snapshot.run_id, expected_sequence=len(current.events) + 1,
                event_ids={previous.event_id for previous in current.events},
            )
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def read(self, run_id: str) -> TraceReadResult:
        with self._open(run_id, write=False) as stream:
            return _read_lines(stream, run_id)