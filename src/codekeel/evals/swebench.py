"""Convert an explicit SWE-bench subset into trusted local eval fixtures."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codekeel.evals.dataset import Task
from codekeel.evals.patch import PatchStatus, export_patch, validate_patch
from codekeel.evals.scorer import EvaluationResult

DatasetId = Literal["SWE-bench/SWE-bench"]
DatasetSplit = Literal["test"]
DATASET_ID: DatasetId = "SWE-bench/SWE-bench"
DATASET_SPLIT: DatasetSplit = "test"

_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODEL_NAME_PATTERN = re.compile(
    r"^codekeel/[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
)
_SENSITIVE_MODEL_PATTERN = re.compile(
    r"(?:api[-_]?key|account|endpoint|password|secret|token|sk-)", re.IGNORECASE
)
_GIT_TIMEOUT = 600.0


class SweBenchConversionError(ValueError):
    """A safe, user-actionable conversion failure."""


class ConversionSummary(BaseModel):
    """Stable CLI result for a completed conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str = DATASET_ID
    revision: str
    split: str = DATASET_SPLIT
    instances: int = Field(ge=1)
    output: str


class PredictionExportSummary(BaseModel):
    """Stable CLI result for a completed predictions export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str = DATASET_ID
    revision: str
    predictions: int = Field(ge=1)
    empty_patches: int = Field(ge=0)
    model_name_or_path: str
    output: str


class _MetadataInstance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    instance_id: str
    repo: str
    base_commit: str
    task_file: str
    repository: str

    @field_validator("instance_id")
    @classmethod
    def valid_instance_id(cls, value: str) -> str:
        return _validated_instance_id(value)

    @field_validator("repo")
    @classmethod
    def valid_repo(cls, value: str) -> str:
        if not _REPO_PATTERN.fullmatch(value):
            raise ValueError("Repository must use the owner/name form")
        owner, name = value.split("/", 1)
        if owner in {".", ".."} or name in {".", ".."}:
            raise ValueError("Repository must use the owner/name form")
        return value

    @field_validator("base_commit")
    @classmethod
    def valid_base_commit(cls, value: str) -> str:
        return _validated_sha(value, "Base commit")

    @model_validator(mode="after")
    def fixed_paths(self):
        if self.task_file != f"{self.instance_id}.yaml":
            raise ValueError("Metadata task path does not match its instance")
        if self.repository != f"repos/{self.instance_id}":
            raise ValueError("Metadata repository path does not match its instance")
        return self


class _ConversionMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1]
    dataset: DatasetId = DATASET_ID
    revision: str
    split: DatasetSplit = DATASET_SPLIT
    codekeel_sha: str
    generated_at: str
    instances: tuple[_MetadataInstance, ...] = Field(min_length=1)

    @field_validator("revision")
    @classmethod
    def valid_revision(cls, value: str) -> str:
        return _validated_sha(value, "Dataset revision")

    @field_validator("codekeel_sha")
    @classmethod
    def valid_codekeel_sha(cls, value: str) -> str:
        return _validated_sha(value, "CodeKeel revision")

    @field_validator("generated_at")
    @classmethod
    def valid_generated_at(cls, value: str) -> str:
        if not value.strip() or "\x00" in value:
            raise ValueError("Generated timestamp is invalid")
        return value

    @model_validator(mode="after")
    def unique_instances(self):
        ids = [item.instance_id for item in self.instances]
        if len(set(ids)) != len(ids):
            raise ValueError("Metadata instance IDs must be unique")
        return self


@dataclass(frozen=True)
class SweBenchInstance:
    """The only SWE-bench fields allowed to cross the conversion boundary."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str


RepositoryMaterializer = Callable[[SweBenchInstance, Path], None]


def convert_swebench_subset(
    instance_ids: Sequence[str], *, revision: str, output: str | Path,
) -> ConversionSummary:
    """Load a fixed remote snapshot and convert only explicitly selected tasks."""
    requested = _validated_requested_ids(instance_ids)
    _validated_output(output)
    resolved_revision, rows = _load_remote_dataset(revision)
    return convert_swebench_records(
        rows,
        requested,
        revision=resolved_revision,
        output=output,
        materialize_repository=_materialize_repository,
        codekeel_sha=_codekeel_sha(),
    )


def convert_swebench_records(
    rows: Iterable[Mapping[str, Any]],
    instance_ids: Sequence[str],
    *,
    revision: str,
    output: str | Path,
    materialize_repository: RepositoryMaterializer,
    codekeel_sha: str,
    generated_at: str | None = None,
) -> ConversionSummary:
    """Convert injected records, keeping network and dataset SDKs outside tests."""
    selected = _select_instances(rows, instance_ids)
    resolved_revision = _validated_sha(revision, "Dataset revision")
    code_sha = _validated_sha(codekeel_sha, "CodeKeel revision")
    target = _validated_output(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))

    try:
        repositories = staging / "repos"
        repositories.mkdir()
        metadata_instances: list[dict[str, str]] = []
        for instance in selected:
            repository = repositories / instance.instance_id
            try:
                materialize_repository(instance, repository)
            except SweBenchConversionError:
                raise
            except Exception:
                raise SweBenchConversionError("Unable to materialize selected repository") from None
            _verify_repository(instance, repository)
            yaml_name = f"{instance.instance_id}.yaml"
            yaml_path = staging / yaml_name
            task = Task.model_validate({
                "id": instance.instance_id,
                "repo": f"repos/{instance.instance_id}",
                "task": instance.problem_statement,
                "verification": {"command": _verification_command(instance.base_commit)},
                "limits": {"max_steps": 40, "timeout": 1800.0},
            })
            yaml_path.write_text(
                yaml.safe_dump(task.model_dump(mode="python"), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            metadata_instances.append({
                "instance_id": instance.instance_id,
                "repo": instance.repo,
                "base_commit": instance.base_commit,
                "task_file": yaml_name,
                "repository": f"repos/{instance.instance_id}",
            })

        metadata = {
            "schema_version": 1,
            "dataset": DATASET_ID,
            "revision": resolved_revision,
            "split": DATASET_SPLIT,
            "codekeel_sha": code_sha,
            "generated_at": generated_at or datetime.now(UTC).isoformat(),
            "instances": metadata_instances,
        }
        (staging / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.replace(target)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return ConversionSummary(
        revision=resolved_revision,
        instances=len(selected),
        output=str(Path(output)),
    )


def export_swebench_predictions(
    dataset: str | Path,
    run: str | Path,
    *,
    model_name: str,
    output: str | Path,
) -> PredictionExportSummary:
    """Combine trusted conversion metadata and verified patch artifacts."""
    dataset_root = _validated_existing_directory(dataset, "Dataset")
    run_root = _validated_existing_directory(run, "Run")
    model = _validated_model_name(model_name)
    target = _validated_prediction_output(output, dataset_root, run_root)

    try:
        metadata = _ConversionMetadata.model_validate(
            _read_json_object(dataset_root / "metadata.json", "Dataset metadata")
        )
        results = _read_results(run_root / "results.jsonl")
    except SweBenchConversionError:
        raise
    except Exception:
        raise SweBenchConversionError("Unable to validate prediction inputs") from None

    expected = {item.instance_id: item for item in metadata.instances}
    actual: dict[str, EvaluationResult] = {}
    for result in results:
        _validated_instance_id(result.task_id)
        if result.task_id in actual:
            raise SweBenchConversionError("Run results contain a duplicate task")
        actual[result.task_id] = result
    if set(actual) - set(expected):
        raise SweBenchConversionError("Run results contain tasks outside the converted dataset")
    if set(expected) - set(actual):
        raise SweBenchConversionError("Run results are missing converted dataset tasks")

    lines: list[str] = []
    empty_patches = 0
    for instance_id in sorted(expected):
        result = actual[instance_id]
        payload = _validated_patch(run_root, instance_id, result)
        empty_patches += result.patch_status is PatchStatus.EMPTY
        prediction = {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": payload.decode("utf-8", "strict"),
        }
        lines.append(json.dumps(prediction, ensure_ascii=False, separators=(",", ":")))
    encoded = ("\n".join(lines) + "\n").encode("utf-8")
    _atomic_write(target, encoded)
    return PredictionExportSummary(
        revision=metadata.revision,
        predictions=len(lines),
        empty_patches=empty_patches,
        model_name_or_path=model,
        output=str(target),
    )


def verify_swebench_patch(repository: str | Path, base_commit: str) -> bool:
    """Return whether the trusted final patch is safe and nonempty."""
    with tempfile.TemporaryDirectory(prefix="codekeel-swebench-verification-") as temporary:
        artifact = export_patch(
            repository,
            base_commit,
            Path(temporary) / "candidate.diff",
        )
        if artifact.status is PatchStatus.EXPORTED:
            assert artifact.path is not None
            validate_patch(repository, base_commit, artifact.path)
    return artifact.status is PatchStatus.EXPORTED


def _validated_existing_directory(value: str | Path, label: str) -> Path:
    try:
        raw = Path(value)
        if "\x00" in str(raw) or raw.is_symlink():
            raise SweBenchConversionError(f"{label} directory is unsafe")
        resolved = raw.resolve(strict=True)
        if not resolved.is_dir():
            raise SweBenchConversionError(f"{label} directory is missing")
        return resolved
    except SweBenchConversionError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise SweBenchConversionError(f"{label} directory is missing or unsafe") from None


def _validated_model_name(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 256
        or not _MODEL_NAME_PATTERN.fullmatch(value)
        or _SENSITIVE_MODEL_PATTERN.search(value)
    ):
        raise SweBenchConversionError("Model name must be a sanitized codekeel identifier")
    return value


def _validated_prediction_output(output: str | Path, dataset: Path, run: Path) -> Path:
    try:
        raw = Path(output)
        if "\x00" in str(raw) or raw.suffix != ".jsonl" or raw.is_symlink():
            raise SweBenchConversionError("Predictions output must be a safe .jsonl file")
        target = raw.resolve(strict=False)
        parent = target.parent.resolve(strict=True)
        if parent.is_symlink() or not parent.is_dir():
            raise SweBenchConversionError("Predictions output directory is unsafe")
        if target.exists() and not target.is_file():
            raise SweBenchConversionError("Predictions output must be a regular file")
        if target in {dataset / "metadata.json", run / "results.jsonl"} or target.is_relative_to(run / "patches"):
            raise SweBenchConversionError("Predictions output overlaps trusted input artifacts")
        return target
    except SweBenchConversionError:
        raise
    except (OSError, RuntimeError, ValueError):
        raise SweBenchConversionError("Predictions output path is invalid") from None


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SweBenchConversionError("JSON input contains duplicate fields")
        value[key] = item
    return value


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file():
            raise SweBenchConversionError(f"{label} is missing or unsafe")
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(value, dict):
            raise SweBenchConversionError(f"{label} must be a JSON object")
        return value
    except SweBenchConversionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SweBenchConversionError(f"{label} is missing, unreadable, or invalid") from None


def _read_results(path: Path) -> tuple[EvaluationResult, ...]:
    try:
        if path.is_symlink() or not path.is_file():
            raise SweBenchConversionError("Run results are missing or unsafe")
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise SweBenchConversionError("Run results must be nonempty JSONL")
        return tuple(
            EvaluationResult.model_validate(
                json.loads(line, object_pairs_hook=_reject_duplicate_json_keys)
            )
            for line in lines
        )
    except SweBenchConversionError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise SweBenchConversionError("Run results are missing, unreadable, or invalid") from None
    except Exception:
        raise SweBenchConversionError("Run results use an invalid schema") from None


def _validated_patch(run: Path, instance_id: str, result: EvaluationResult) -> bytes:
    if result.patch_status is PatchStatus.ERROR:
        raise SweBenchConversionError("Run contains a patch export error")
    if result.patch_path is None or result.patch_sha256 is None or result.patch_bytes is None:
        raise SweBenchConversionError("Run result is missing patch identity fields")
    expected = run / "patches" / f"{instance_id}.diff"
    if result.patch_path != str(expected) or expected.parent.is_symlink() or expected.is_symlink():
        raise SweBenchConversionError("Patch path does not match its trusted run artifact")
    try:
        if not expected.is_file() or not stat.S_ISREG(expected.lstat().st_mode):
            raise SweBenchConversionError("Patch is missing or unsafe")
        descriptor = os.open(expected, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise SweBenchConversionError("Patch is missing or unsafe")
            payload = stream.read()
        payload.decode("utf-8", "strict")
    except SweBenchConversionError:
        raise
    except (OSError, UnicodeError):
        raise SweBenchConversionError("Patch is missing, unreadable, or not UTF-8") from None
    digest = hashlib.sha256(payload).hexdigest()
    if not _SHA256_PATTERN.fullmatch(result.patch_sha256) or digest != result.patch_sha256:
        raise SweBenchConversionError("Patch SHA256 does not match the run result")
    if len(payload) != result.patch_bytes:
        raise SweBenchConversionError("Patch size does not match the run result")
    if (result.patch_status is PatchStatus.EMPTY) != (not payload):
        raise SweBenchConversionError("Patch status does not match its content")
    return payload


def _atomic_write(target: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}-", dir=target.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(target)
        temporary = None
    except OSError:
        raise SweBenchConversionError("Unable to write predictions output") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _load_remote_dataset(revision: str) -> tuple[str, Iterable[Mapping[str, Any]]]:
    if not isinstance(revision, str) or not revision.strip() or "\x00" in revision:
        raise SweBenchConversionError("Dataset revision must be nonblank and contain no NUL")
    try:
        datasets = importlib.import_module("datasets")
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as error:
        raise SweBenchConversionError(
            "SWE-bench conversion requires: uv sync --group swebench"
        ) from error
    try:
        info = huggingface_hub.HfApi().dataset_info(repo_id=DATASET_ID, revision=revision)
        resolved = _validated_sha(info.sha, "Resolved dataset revision")
        rows = datasets.load_dataset(DATASET_ID, revision=resolved, split=DATASET_SPLIT)
    except SweBenchConversionError:
        raise
    except Exception as error:
        raise SweBenchConversionError("Unable to load the fixed SWE-bench dataset snapshot") from error
    return resolved, rows


def _select_instances(
    rows: Iterable[Mapping[str, Any]], instance_ids: Sequence[str],
) -> tuple[SweBenchInstance, ...]:
    requested = _validated_requested_ids(instance_ids)
    wanted = set(requested)
    matches: dict[str, SweBenchInstance] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise SweBenchConversionError("Dataset rows must be mappings")
        raw_id = row.get("instance_id")
        if not isinstance(raw_id, str):
            continue
        if raw_id not in wanted:
            continue
        if raw_id in matches:
            raise SweBenchConversionError("Selected instance occurs more than once")
        matches[raw_id] = _project_instance(row)
    if set(matches) != wanted:
        raise SweBenchConversionError("One or more selected instances were not found")
    return tuple(matches[item] for item in sorted(matches))


def _validated_requested_ids(instance_ids: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(instance_ids)
    if not requested:
        raise SweBenchConversionError("At least one instance ID is required")
    for instance_id in requested:
        _validated_instance_id(instance_id)
    if len(set(requested)) != len(requested):
        raise SweBenchConversionError("Instance IDs must be unique")
    return requested


def _project_instance(row: Mapping[str, Any]) -> SweBenchInstance:
    instance_id = _validated_instance_id(row.get("instance_id"))
    repo = row.get("repo")
    if not isinstance(repo, str) or not _REPO_PATTERN.fullmatch(repo):
        raise SweBenchConversionError("Repository must use the owner/name form")
    owner, name = repo.split("/", 1)
    if owner in {".", ".."} or name in {".", ".."}:
        raise SweBenchConversionError("Repository must use the owner/name form")
    base_commit = _validated_sha(row.get("base_commit"), "Base commit")
    problem = row.get("problem_statement")
    if not isinstance(problem, str) or not problem.strip() or "\x00" in problem:
        raise SweBenchConversionError("Problem statement must be nonblank and contain no NUL")
    return SweBenchInstance(instance_id, repo, base_commit, problem)


def _validated_instance_id(value: object) -> str:
    if not isinstance(value, str) or not _ID_PATTERN.fullmatch(value):
        raise SweBenchConversionError("Instance ID is invalid")
    return value


def _validated_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA_PATTERN.fullmatch(value):
        raise SweBenchConversionError(f"{label} must be a 40-character hexadecimal SHA")
    return value.lower()


def _validated_output(output: str | Path) -> Path:
    try:
        raw = Path(output)
        if "\x00" in str(raw):
            raise SweBenchConversionError("Output path contains a NUL")
        if raw.is_symlink():
            raise SweBenchConversionError("Output must be a directory, not a link or file")
        target = raw.resolve(strict=False)
        if target == Path(target.anchor):
            raise SweBenchConversionError("Output must not be a filesystem root")
        if target.exists():
            mode = target.lstat().st_mode
            if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
                raise SweBenchConversionError("Output must be a directory, not a link or file")
            if any(target.iterdir()):
                raise SweBenchConversionError("Output directory must be empty")
        return target
    except SweBenchConversionError:
        raise
    except (OSError, ValueError) as error:
        raise SweBenchConversionError("Output path is invalid or unreadable") from error


def _materialize_repository(instance: SweBenchInstance, destination: Path) -> None:
    url = f"https://github.com/{instance.repo}.git"
    _run_git(["init", "--quiet", str(destination)])
    _run_git(["-C", str(destination), "remote", "add", "origin", url])
    _run_git([
        "-C", str(destination), "fetch", "--quiet", "--depth=1", "origin", instance.base_commit,
    ])
    _run_git(["-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"])


def _verify_repository(instance: SweBenchInstance, repository: Path) -> None:
    if not repository.is_dir() or repository.is_symlink():
        raise SweBenchConversionError("Materialized repository is missing or unsafe")
    head = _run_git(["-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip()
    if head.lower() != instance.base_commit:
        raise SweBenchConversionError("Materialized repository HEAD does not match base commit")
    status_result = _run_git([
        "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all",
    ])
    if status_result.stdout:
        raise SweBenchConversionError("Materialized repository must start clean")
    for entry in repository.rglob("*"):
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode) or not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise SweBenchConversionError("Materialized repository contains a link or special file")


def _run_git(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update({
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_TERMINAL_PROMPT": "0",
    })
    command = ["git", "-c", f"core.hooksPath={os.devnull}", *arguments]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            check=False,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise SweBenchConversionError("Unable to prepare or inspect repository") from error
    if result.returncode != 0:
        raise SweBenchConversionError("Unable to prepare or inspect repository")
    return result


def _verification_command(base_commit: str) -> str:
    return f"codekeel eval-verify-swebench-patch --base-commit {base_commit}"


def _codekeel_sha() -> str:
    repository = Path(__file__).resolve().parents[3]
    return _validated_sha(
        _run_git(["-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"]).stdout.strip(),
        "CodeKeel revision",
    )