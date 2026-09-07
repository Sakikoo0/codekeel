"""Storage boundary with compare-and-swap revisions for checkpoint ownership."""

from threading import RLock
from typing import Protocol

from codekeel.events.store import validate_run_id
from codekeel.persistence.checkpoint import Checkpoint


class PersistenceError(RuntimeError):
    """Checkpoint storage failed or a stale runtime attempted to write."""


class ResumeError(PersistenceError):
    """The checkpoint is not a safe continuation point."""


class CheckpointStore(Protocol):
    def load(self, run_id: str) -> Checkpoint: ...

    def save(self, checkpoint: Checkpoint, *, expected_revision: int) -> None:
        """Atomically save revision expected_revision + 1; zero creates a run."""
        ...


class MemoryCheckpointStore:
    def __init__(self) -> None:
        self._runs: dict[str, str] = {}
        self._lock = RLock()

    def load(self, run_id: str) -> Checkpoint:
        validate_run_id(run_id)
        with self._lock:
            if run_id not in self._runs:
                raise FileNotFoundError("Checkpoint not found")
            return Checkpoint.model_validate_json(self._runs[run_id])

    def save(self, checkpoint: Checkpoint, *, expected_revision: int) -> None:
        snapshot = Checkpoint.model_validate_json(checkpoint.model_dump_json())
        with self._lock:
            current = self.load(snapshot.run_id).revision if snapshot.run_id in self._runs else 0
            if current != expected_revision or snapshot.revision != expected_revision + 1:
                raise PersistenceError("Checkpoint revision conflict")
            self._runs[snapshot.run_id] = snapshot.model_dump_json()