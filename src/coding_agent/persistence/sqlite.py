"""SQLite checkpoints below a trusted host root, separate from append-only traces."""

import os
import sqlite3
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from coding_agent.events.store import validate_run_id
from coding_agent.persistence.checkpoint import Checkpoint
from coding_agent.persistence.store import PersistenceError


class SqliteCheckpointStore:
    """Short transactions and revision CAS; no model calls occur in a transaction.

    The root is a trusted host location, preferably outside the tool workspace.
    Existing symlink components, linked database files and unsafe sidecars fail closed.
    """

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root).resolve()

    @contextmanager
    def _connect(self, *, create: bool) -> Iterator[sqlite3.Connection]:
        directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        descriptor = None
        connection = None
        try:
            if create:
                try:
                    os.mkdir(".agent", mode=0o700, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(".agent", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
            name = "checkpoints.sqlite3"
            flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK
            if create:
                flags |= os.O_CREAT
            descriptor = os.open(name, flags, 0o600, dir_fd=directory)
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise PersistenceError("Checkpoint database must be a regular file with one link")
            for suffix in ("-journal", "-wal", "-shm"):
                try:
                    sidecar = os.stat(name + suffix, dir_fd=directory, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if not stat.S_ISREG(sidecar.st_mode) or sidecar.st_nlink != 1:
                    raise PersistenceError("Unsafe SQLite sidecar")
            path = self.root / ".agent" / name
            connection = sqlite3.connect(path.as_uri() + "?mode=rw", uri=True, timeout=5)
            current = path.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (metadata.st_dev, metadata.st_ino):
                raise PersistenceError("Checkpoint database changed while opening")
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA synchronous=FULL")
            if create:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT PRIMARY KEY, revision INTEGER NOT NULL, "
                    "payload TEXT NOT NULL)"
                )
            yield connection
        except sqlite3.Error as error:
            raise PersistenceError("Unable to access checkpoint database") from error
        finally:
            if connection is not None:
                connection.close()
            if descriptor is not None:
                os.close(descriptor)
            os.close(directory)

    def load(self, run_id: str) -> Checkpoint:
        validate_run_id(run_id)
        with self._connect(create=False) as connection:
            row = connection.execute("SELECT revision, payload FROM checkpoints WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise FileNotFoundError("Checkpoint not found")
        try:
            snapshot = Checkpoint.model_validate_json(row[1])
        except ValueError as error:
            raise PersistenceError("Invalid checkpoint payload") from error
        if snapshot.run_id != run_id or snapshot.revision != row[0]:
            raise PersistenceError("Checkpoint identity mismatch")
        return snapshot

    def save(self, checkpoint: Checkpoint, *, expected_revision: int) -> None:
        snapshot = Checkpoint.model_validate_json(checkpoint.model_dump_json())
        if snapshot.revision != expected_revision + 1:
            raise PersistenceError("Checkpoint revision conflict")
        with self._connect(create=True) as connection, connection:
            if expected_revision == 0:
                connection.execute("INSERT INTO checkpoints VALUES (?, ?, ?)",
                                   (snapshot.run_id, snapshot.revision, snapshot.model_dump_json()))
            else:
                updated = connection.execute(
                    "UPDATE checkpoints SET revision = ?, payload = ? WHERE run_id = ? AND revision = ?",
                    (snapshot.revision, snapshot.model_dump_json(), snapshot.run_id, expected_revision),
                )
                if updated.rowcount != 1:
                    raise PersistenceError("Checkpoint revision conflict")