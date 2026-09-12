"""Trusted Git patch export for disposable evaluation workspaces."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath

_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_GIT_TIMEOUT = 120.0
_IGNORED_COMPONENTS = frozenset({
    ".agent", ".mypy_cache", ".nox", ".pytest_cache", ".ruff_cache", ".tox", ".venv",
    "__pycache__", "evaluator-logs", "htmlcov", "logs",
})
_IGNORED_NAMES = frozenset({".coverage"})
_IGNORED_SUFFIXES = (".log", ".pyc", ".pyo")
_SENSITIVE_COMPONENTS = frozenset({".aws", ".ssh"})
_SENSITIVE_NAMES = frozenset({
    "credentials.json", "evaluator-report.json", "patch.diff", "predictions.jsonl", "secrets.json", "token.json",
})
_SENSITIVE_SUFFIXES = (".key", ".p12", ".pem", ".pfx")


class PatchStatus(StrEnum):
    """Mutually exclusive patch export outcomes."""

    EXPORTED = "exported"
    EMPTY = "empty"
    ERROR = "error"


@dataclass(frozen=True)
class PatchArtifact:
    """Persisted patch identity passed into evaluation accounting."""

    status: PatchStatus
    path: str | None = None
    sha256: str | None = None
    size: int | None = None

    @classmethod
    def error(cls) -> PatchArtifact:
        return cls(PatchStatus.ERROR)


class PatchExportError(RuntimeError):
    """A sanitized failure to capture or persist an evaluation patch."""


def capture_base_commit(repository: str | Path) -> str:
    """Capture HEAD only when repository is the exact Git worktree root."""
    root = _validated_repository(repository)
    top_level = os.fsdecode(_run_git(root, ["rev-parse", "--show-toplevel"]).stdout.strip())
    try:
        if Path(top_level).resolve(strict=True) != root:
            raise PatchExportError("Evaluation workspace must be a Git worktree root")
    except OSError:
        raise PatchExportError("Unable to resolve the Git worktree root") from None
    commit = _run_git(root, ["rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip().decode("ascii", "strict")
    if not _SHA_PATTERN.fullmatch(commit):
        raise PatchExportError("Evaluation workspace HEAD is not a commit")
    return commit.lower()


def export_patch(repository: str | Path, base_commit: str, destination: str | Path) -> PatchArtifact:
    """Export the final tree relative to base, including safe untracked files."""
    root = _validated_repository(repository)
    base = _validated_base_commit(base_commit)
    target = _validated_destination(destination)
    temporary: Path | None = None
    try:
        _run_git(root, ["cat-file", "-e", f"{base}^{{commit}}"])
        _validate_workspace_entries(root)
        tracked = _changed_paths(
            _run_git(root, [
                "diff", "--no-renames", "--no-ext-diff", "--no-textconv", "--name-only", "-z", base, "--",
            ]).stdout
        )
        untracked = _changed_paths(
            _run_git(root, ["ls-files", "--others", "--exclude-standard", "-z", "--"]).stdout
        )
        safe_tracked = _safe_paths(root, tracked)
        safe_untracked = _safe_paths(root, untracked)

        pieces: list[bytes] = []
        if safe_tracked:
            pieces.append(_run_git(root, [
                "diff", "--binary", "--full-index", "--find-renames", "--no-ext-diff", "--no-textconv",
                "--src-prefix=a/", "--dst-prefix=b/", base, "--", *safe_tracked,
            ]).stdout)
        for path in safe_untracked:
            pieces.append(_run_git(root, [
                "diff", "--no-index", "--binary", "--full-index", "--no-ext-diff", "--no-textconv",
                "--src-prefix=a/", "--dst-prefix=b/", "--", os.devnull, path,
            ], allowed_returncodes=(0, 1)).stdout)
        payload = b"".join(pieces)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        temporary = None
    except PatchExportError:
        raise
    except (OSError, UnicodeError, subprocess.SubprocessError):
        raise PatchExportError("Unable to export evaluation patch") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass

    digest = hashlib.sha256(payload).hexdigest()
    status = PatchStatus.EXPORTED if payload else PatchStatus.EMPTY
    return PatchArtifact(status=status, path=str(target), sha256=digest, size=len(payload))


def validate_patch(repository: str | Path, base_commit: str, patch: str | Path) -> None:
    """Check an exported patch against base without changing the real index or tree."""
    root = _validated_repository(repository)
    base = _validated_base_commit(base_commit)
    try:
        candidate = Path(patch)
        if candidate.is_symlink():
            raise PatchExportError("Patch validation input must be a regular file")
        candidate = candidate.resolve(strict=True)
        if not stat.S_ISREG(candidate.lstat().st_mode):
            raise PatchExportError("Patch validation input must be a regular file")
    except PatchExportError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PatchExportError("Patch validation input is missing or unsafe") from None

    with tempfile.TemporaryDirectory(prefix="codekeel-patch-index-") as temporary:
        index = Path(temporary) / "index"
        _run_git(root, ["read-tree", base], index_file=index)
        _run_git(
            root,
            ["apply", "--cached", "--check", "--whitespace=error-all", "--", str(candidate)],
            index_file=index,
        )


def _validated_repository(repository: str | Path) -> Path:
    try:
        raw = Path(repository)
        if raw.is_symlink():
            raise PatchExportError("Evaluation workspace must not be a symbolic link")
        root = raw.resolve(strict=True)
    except PatchExportError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PatchExportError("Evaluation workspace is missing or unsafe") from None
    if not root.is_dir():
        raise PatchExportError("Evaluation workspace is not a directory")
    return root


def _validated_base_commit(value: object) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise PatchExportError("Base commit must be a 40-character hexadecimal SHA")
    return value.lower()


def _validated_destination(destination: str | Path) -> Path:
    try:
        raw = Path(destination)
        if "\x00" in str(raw) or raw.is_symlink() or raw.exists():
            raise PatchExportError("Patch destination must be a new regular file")
        target = raw.resolve(strict=False)
        if target == Path(target.anchor):
            raise PatchExportError("Patch destination must not be a filesystem root")
        parent = target.parent
        if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
            raise PatchExportError("Patch destination parent is unsafe")
        return target
    except PatchExportError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise PatchExportError("Patch destination is invalid") from None


def _changed_paths(output: bytes) -> tuple[str, ...]:
    if not output:
        return ()
    if not output.endswith(b"\0"):
        raise PatchExportError("Git returned a malformed path list")
    paths = tuple(os.fsdecode(item) for item in output[:-1].split(b"\0"))
    if any(not path for path in paths):
        raise PatchExportError("Git returned an empty changed path")
    return paths


def _safe_paths(root: Path, paths: Sequence[str]) -> tuple[str, ...]:
    safe: list[str] = []
    for path in paths:
        policy = _path_policy(path)
        if policy == "ignore":
            continue
        if policy == "reject":
            raise PatchExportError("Patch contains a sensitive or benchmark-private path")
        _validate_workspace_path(root, path)
        safe.append(path)
    return tuple(sorted(set(safe), key=os.fsencode))


def _path_policy(path: str) -> str:
    logical = PurePosixPath(path)
    if logical.is_absolute() or not logical.parts or any(part in {"", ".", ".."} for part in logical.parts):
        raise PatchExportError("Git returned an unsafe changed path")
    lowered = tuple(part.lower() for part in logical.parts)
    name = lowered[-1]
    if any(part in _IGNORED_COMPONENTS for part in lowered):
        return "ignore"
    if name in _IGNORED_NAMES or name.endswith(_IGNORED_SUFFIXES):
        return "ignore"
    if any(part in _SENSITIVE_COMPONENTS for part in lowered):
        return "reject"
    if name == ".env" or name.startswith(".env."):
        return "reject"
    if name in _SENSITIVE_NAMES or name.endswith(_SENSITIVE_SUFFIXES):
        return "reject"
    return "include"


def _validate_workspace_path(root: Path, relative: str) -> None:
    current = root
    parts = PurePosixPath(relative).parts
    for index, part in enumerate(parts):
        current /= part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            return
        except OSError:
            raise PatchExportError("Unable to inspect a changed path") from None
        if stat.S_ISLNK(mode):
            raise PatchExportError("Patch paths must not contain symbolic links")
        if index < len(parts) - 1 and not stat.S_ISDIR(mode):
            raise PatchExportError("Patch path parent is not a directory")
        if index == len(parts) - 1 and not stat.S_ISREG(mode):
            raise PatchExportError("Patch contains a special file")


def _validate_workspace_entries(root: Path) -> None:
    """Reject filesystem objects that Git path enumeration cannot represent."""
    pending: list[tuple[Path, tuple[str, ...]]] = [(root, ())]
    while pending:
        directory, parent_parts = pending.pop()
        try:
            entries = list(os.scandir(directory))
        except OSError:
            raise PatchExportError("Unable to inspect workspace entries") from None
        for entry in entries:
            parts = (*parent_parts, entry.name)
            if parts == (".git",):
                continue
            relative = PurePosixPath(*parts).as_posix()
            if _path_policy(relative) == "ignore":
                continue
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError:
                raise PatchExportError("Unable to inspect workspace entries") from None
            if stat.S_ISLNK(mode):
                raise PatchExportError("Workspace contains a symbolic link")
            if stat.S_ISDIR(mode):
                pending.append((Path(entry.path), parts))
            elif not stat.S_ISREG(mode):
                raise PatchExportError("Workspace contains a special file")


def _run_git(
    repository: Path,
    arguments: Sequence[str],
    *,
    allowed_returncodes: Sequence[int] = (0,),
    index_file: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_LITERAL_PATHSPECS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
    })
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    command = [
        "git", "-c", f"core.hooksPath={os.devnull}", "-c", "core.fsmonitor=false",
        "-c", "core.quotePath=true", *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            cwd=repository,
            capture_output=True,
            timeout=_GIT_TIMEOUT,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError):
        raise PatchExportError("Unable to inspect Git workspace") from None
    if result.returncode not in allowed_returncodes:
        raise PatchExportError("Unable to inspect Git workspace")
    return result