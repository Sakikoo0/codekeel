"""Trusted local YAML tasks; repository paths are relative to each task file."""

import stat
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    max_steps: int = Field(default=20, gt=0)
    timeout: float = Field(default=300.0, gt=0)


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    command: str = Field(min_length=1, max_length=4096)

    @field_validator("command")
    @classmethod
    def valid_command(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Command must be nonblank and contain no NUL")
        return value


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
    repo: str = Field(min_length=1)
    task: str = Field(min_length=1)
    verification: Verification
    limits: TaskLimits = Field(default_factory=TaskLimits)

    @field_validator("repo")
    @classmethod
    def relative_repo(cls, value: str) -> str:
        if Path(value).is_absolute() or ".." in Path(value).parts or "\x00" in value or not value.strip():
            raise ValueError("Repository must be a contained relative path")
        return value

    @field_validator("task")
    @classmethod
    def valid_task(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Task must be nonblank and contain no NUL")
        return value


@dataclass(frozen=True)
class DatasetTask:
    task: Task
    source: Path

    def repository(self) -> Path:
        base = self.source.parent.resolve(strict=True)
        candidate = base
        for part in Path(self.task.repo).parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("Repository path contains a symbolic link")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(base) or not resolved.is_dir():
            raise ValueError("Repository must be a contained directory")
        return resolved


def load_dataset(path: str | Path) -> list[DatasetTask]:
    """Load all immediate YAML files in lexical order, rejecting duplicate IDs.

    Dataset commands are trusted executable configuration, not model input.
    No remote repositories, setup hooks, or arbitrary YAML object constructors.
    """
    path = Path(path)
    files = sorted((*path.glob("*.yaml"), *path.glob("*.yml"))) if path.is_dir() else [path]
    if not files:
        raise ValueError("Dataset has no YAML tasks")
    tasks: list[DatasetTask] = []
    ids: set[str] = set()
    for source in files:
        if not stat.S_ISREG(source.lstat().st_mode) or source.stat().st_size > 1_000_000:
            raise ValueError("Task must be a regular YAML file of at most 1 MB")
        task = Task.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
        if task.id in ids:
            raise ValueError("Duplicate task ID")
        ids.add(task.id)
        entry = DatasetTask(task, source.resolve())
        entry.repository()
        tasks.append(entry)
    return tasks