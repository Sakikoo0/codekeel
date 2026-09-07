"""Run identical tasks under independent configs and collect comparison metrics."""

import os
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codekeel.evals.config import ExperimentConfig
from codekeel.evals.dataset import DatasetTask
from codekeel.evals.runner import run_dataset
from codekeel.evals.scorer import EvaluationResult
from codekeel.models.base import Model
from codekeel.workspace.base import Workspace
from codekeel.workspace.local import LocalWorkspace


class Summary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    tasks: int = Field(gt=0)
    successes: int = Field(ge=0)
    success_rate: float = Field(ge=0, le=1)
    avg_tokens: float = Field(ge=0)
    avg_steps: float = Field(ge=0)
    avg_cost: float = Field(ge=0)
    avg_duration: float = Field(ge=0)


def aggregate(results: Sequence[EvaluationResult]) -> Summary:
    """All tasks, including errors and timeouts, contribute to every mean."""
    count = len(results)
    if not count or len({item.task_id for item in results}) != count:
        raise ValueError("Aggregation requires nonempty results with unique task IDs")
    successes = sum(item.success for item in results)
    return Summary(
        tasks=count, successes=successes, success_rate=successes / count,
        avg_tokens=sum(item.input_tokens + item.output_tokens for item in results) / count,
        avg_steps=sum(item.steps for item in results) / count,
        avg_cost=sum(item.cost for item in results) / count,
        avg_duration=sum(item.duration for item in results) / count,
    )


class ConfigurationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    config: ExperimentConfig
    results: tuple[EvaluationResult, ...]
    summary: Summary


class ComparisonReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    configurations: tuple[ConfigurationResult, ...]

    def markdown(self) -> str:
        lines = [
            "| Config | Success | Tokens | Steps | Cost | Duration |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for group in self.configurations:
            s = group.summary
            lines.append(
                f"| {group.config.name} | {s.successes}/{s.tasks} ({s.success_rate:.1%}) | "
                f"{s.avg_tokens:.2f} | {s.avg_steps:.2f} | {s.avg_cost:.6f} | {s.avg_duration:.3f}s |"
            )
        return "\n".join(lines) + "\n\nTokens, steps, cost and duration are per-task means, including failures.\n"


@dataclass(frozen=True)
class ExperimentRun:
    directory: Path
    report: ComparisonReport


def _save_report(directory: Path, report: ComparisonReport) -> None:
    # The directory was exclusively created by this invocation. Replace complete
    # snapshots so interruption retains a valid report for finished configs.
    for name, content in (("results.json", report.model_dump_json(indent=2) + "\n"),
                          ("summary.md", report.markdown())):
        temporary = directory / (name + ".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(directory / name)


async def run_experiments(
    tasks: Sequence[DatasetTask], configs: Sequence[ExperimentConfig], *, model_factory: Callable[[], Model],
    root: str | Path = ".agent/evals", workspace_factory: Callable[[Path], Workspace] = LocalWorkspace,
) -> ExperimentRun:
    """Each config/task pair reuses the isolated runner with fresh dependencies."""
    tasks, configs = tuple(tasks), tuple(configs)
    if not tasks or len({task.task.id for task in tasks}) != len(tasks):
        raise ValueError("Comparison requires tasks with unique IDs")
    if not configs or len({config.name for config in configs}) != len(configs):
        raise ValueError("Comparison requires configs with unique names")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="eval-", dir=root))
    groups: list[ConfigurationResult] = []
    for config in configs:
        evaluation = await run_dataset(tasks, model_factory=model_factory, workspace_factory=workspace_factory,
                                       root=directory, config=config)
        groups.append(ConfigurationResult(config=config, results=evaluation.results,
                                          summary=aggregate(evaluation.results)))
        _save_report(directory, ComparisonReport(configurations=tuple(groups)))
    return ExperimentRun(directory, ComparisonReport(configurations=tuple(groups)))
