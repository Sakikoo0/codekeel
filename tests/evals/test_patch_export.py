"""Trusted patch export and runner integration tests."""

import hashlib
import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest
import yaml

from codekeel.evals import patch as patch_module
from codekeel.evals.dataset import load_dataset
from codekeel.evals.patch import (
    PatchExportError,
    PatchStatus,
    capture_base_commit,
    export_patch,
)
from codekeel.evals.runner import run_dataset
from codekeel.models.base import ModelResponse, ToolCall
from codekeel.models.fake import FakeModel
from codekeel.workspace.local import LocalWorkspace


def git(repository: Path, *arguments: str, allowed=(0,)) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in allowed:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


def initialize_repository(path: Path, files: dict[str, bytes] | None = None) -> str:
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.email", "tests@example.invalid")
    git(path, "config", "user.name", "CodeKeel Tests")
    for name, content in (files or {"module.py": b"VALUE = 1\n"}).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    git(path, "add", ".")
    git(path, "commit", "--quiet", "-m", "base")
    return git(path, "rev-parse", "HEAD")


def test_exports_commits_worktree_changes_renames_modes_and_untracked_files(tmp_path):
    repository = tmp_path / "repository"
    base = initialize_repository(repository, {
        "committed.py": b"BEFORE = 1\n",
        "delete.py": b"DELETE = True\n",
        "mode.sh": b"#!/bin/sh\necho mode\n",
        "modify.py": b"VALUE = 1\n",
        "old.py": b"RENAMED = True\n",
    })
    (repository / "committed.py").write_text("AFTER = 2\n")
    git(repository, "add", "committed.py")
    git(repository, "commit", "--quiet", "-m", "agent commit")
    (repository / "modify.py").write_text("VALUE = 2\n")
    (repository / "delete.py").unlink()
    git(repository, "mv", "old.py", "renamed.py")
    (repository / "mode.sh").chmod(0o755)
    (repository / "nested").mkdir()
    (repository / "nested" / "new.py").write_text("NEW = True\n")
    (repository / "binary.bin").write_bytes(b"\x00\x01\x02binary")
    (repository / "empty.txt").touch()
    unusual = repository / ":(exclude)*\nname.py"
    unusual.write_text("ODD = True\n")

    destination = tmp_path / "artifacts" / "task.diff"
    artifact = export_patch(repository, base, destination)

    assert artifact.status is PatchStatus.EXPORTED
    assert artifact.path == str(destination.resolve())
    assert artifact.size == destination.stat().st_size > 0
    assert artifact.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    patch = destination.read_bytes()
    assert b"GIT binary patch" in patch and b"new file mode 100644" in patch

    applied = tmp_path / "applied"
    shutil.copytree(repository, applied)
    git(applied, "reset", "--hard", base)
    git(applied, "clean", "-fd")
    git(applied, "apply", "--binary", str(destination))
    assert (applied / "committed.py").read_text() == "AFTER = 2\n"
    assert (applied / "modify.py").read_text() == "VALUE = 2\n"
    assert not (applied / "delete.py").exists() and not (applied / "old.py").exists()
    assert (applied / "renamed.py").read_text() == "RENAMED = True\n"
    assert (applied / "binary.bin").read_bytes() == b"\x00\x01\x02binary"
    assert (applied / "nested" / "new.py").read_text() == "NEW = True\n"
    assert (applied / "empty.txt").read_bytes() == b""
    assert (applied / unusual.name).read_text() == "ODD = True\n"
    assert (applied / "mode.sh").stat().st_mode & stat.S_IXUSR


def test_empty_patch_is_persisted_with_distinct_status(tmp_path):
    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    destination = tmp_path / "empty.diff"

    artifact = export_patch(repository, base, destination)

    assert artifact.status is PatchStatus.EMPTY
    assert artifact.path == str(destination.resolve())
    assert artifact.size == 0 and destination.read_bytes() == b""
    assert artifact.sha256 == hashlib.sha256(b"").hexdigest()


def test_runtime_files_are_excluded_from_patch(tmp_path):
    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    generated = {
        ".agent/private.json": b"secret",
        ".venv/bin/tool": b"generated",
        "__pycache__/module.pyc": b"generated",
        ".pytest_cache/state": b"generated",
        "logs/task.log": b"generated",
    }
    for name, content in generated.items():
        target = repository / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    artifact = export_patch(repository, base, tmp_path / "ignored.diff")

    assert artifact.status is PatchStatus.EMPTY
    assert Path(artifact.path).read_bytes() == b""


@pytest.mark.parametrize("name", [".env", ".env.local", "credentials.json", "private.pem", ".ssh/id_rsa"])
def test_sensitive_or_private_files_fail_closed(tmp_path, name):
    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    target = repository / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("API_KEY=secret")
    destination = tmp_path / "patch.diff"

    with pytest.raises(PatchExportError, match="sensitive") as failure:
        export_patch(repository, base, destination)
    assert "API_KEY" not in str(failure.value)
    assert not destination.exists()


@pytest.mark.parametrize("kind", ["symlink", "fifo"])
def test_links_and_special_files_cannot_enter_patch(tmp_path, kind):
    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    unsafe = repository / "unsafe"
    if kind == "symlink":
        unsafe.symlink_to(repository / "module.py")
    else:
        os.mkfifo(unsafe)

    with pytest.raises(PatchExportError, match="symbolic|special"):
        export_patch(repository, base, tmp_path / "patch.diff")


def test_external_diff_textconv_and_hooks_are_not_executed(tmp_path):
    repository = tmp_path / "repository"
    sentinel = tmp_path / "PWNED"
    script = tmp_path / "malicious.sh"
    script.write_text(f"#!/bin/sh\ntouch {sentinel}\n")
    script.chmod(0o755)
    base = initialize_repository(repository, {".gitattributes": b"*.txt diff=evil\n", "data.txt": b"before\n"})
    git(repository, "config", "diff.external", str(script))
    git(repository, "config", "diff.evil.textconv", str(script))
    git(repository, "config", "core.fsmonitor", str(script))
    hook = repository / ".git" / "hooks" / "post-checkout"
    hook.write_text(f"#!/bin/sh\ntouch {sentinel}\n")
    hook.chmod(0o755)
    (repository / "data.txt").write_text("after\n")

    artifact = export_patch(repository, base, tmp_path / "safe.diff")

    assert artifact.status is PatchStatus.EXPORTED
    assert not sentinel.exists()
    assert b"-before" in Path(artifact.path).read_bytes() and b"+after" in Path(artifact.path).read_bytes()


@pytest.mark.parametrize("path", ["../escape.py", "/absolute.py", "nested/../../escape.py"])
def test_untrusted_git_paths_cannot_escape_workspace(tmp_path, path):
    repository = tmp_path / "repository"
    initialize_repository(repository)

    with pytest.raises(PatchExportError, match="unsafe"):
        patch_module._safe_paths(repository, (path,))


def test_unreadable_untracked_file_fails_without_partial_artifact(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    (repository / "unreadable.bin").write_bytes(b"new")
    destination = tmp_path / "patch.diff"
    real_run = patch_module.subprocess.run

    def unreadable_file(command, **kwargs):
        if "--no-index" in command:
            raise PermissionError("API_KEY=secret")
        return real_run(command, **kwargs)

    monkeypatch.setattr(patch_module.subprocess, "run", unreadable_file)
    with pytest.raises(PatchExportError) as failure:
        export_patch(repository, base, destination)
    assert "API_KEY" not in str(failure.value)
    assert not destination.exists()


def test_non_repository_wrong_base_and_existing_destination_are_errors(tmp_path):
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(PatchExportError):
        capture_base_commit(plain)

    repository = tmp_path / "repository"
    base = initialize_repository(repository)
    with pytest.raises(PatchExportError):
        export_patch(repository, "not-a-sha", tmp_path / "invalid.diff")
    with pytest.raises(PatchExportError):
        export_patch(repository, "f" * 40, tmp_path / "missing.diff")
    existing = tmp_path / "existing.diff"
    existing.write_text("unchanged")
    with pytest.raises(PatchExportError, match="new regular"):
        export_patch(repository, base, existing)
    assert existing.read_text() == "unchanged"


def write_task(directory: Path, identifier: str, repository: str) -> None:
    (directory / f"{identifier}.yaml").write_text(yaml.safe_dump({
        "id": identifier,
        "repo": repository,
        "task": "Change VALUE to 2.",
        "verification": {"command": "echo checked"},
        "limits": {"max_steps": 3, "timeout": 10.0},
    }))


def repair_model():
    return FakeModel([
        ModelResponse(tool_calls=[ToolCall(
            id="repair", name="write_file", arguments={"path": "module.py", "content": "VALUE = 2\n"},
        )]),
        ModelResponse(content="fixed"),
    ])


async def test_runner_exports_before_close_and_temporary_cleanup(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    repository = dataset / "repo"
    initialize_repository(repository)
    write_task(dataset, "git-task", "repo")
    workspaces = []

    class ObservedWorkspace(LocalWorkspace):
        closed = False

        async def close(self):
            self.closed = True

    def workspace_factory(path):
        workspace = ObservedWorkspace(path)
        workspaces.append(workspace)
        return workspace

    from codekeel.evals import runner

    real_export = runner.export_patch

    def ordered_export(*args, **kwargs):
        assert workspaces and not workspaces[-1].closed
        return real_export(*args, **kwargs)

    monkeypatch.setattr(runner, "export_patch", ordered_export)
    evaluation = await run_dataset(
        load_dataset(dataset), model_factory=repair_model,
        workspace_factory=workspace_factory, root=tmp_path / "results",
    )

    result = evaluation.results[0]
    assert result.success and result.patch_status is PatchStatus.EXPORTED
    assert result.patch_path and Path(result.patch_path).is_file()
    assert result.patch_sha256 == hashlib.sha256(Path(result.patch_path).read_bytes()).hexdigest()
    assert result.patch_bytes == Path(result.patch_path).stat().st_size
    assert workspaces[0].closed and not workspaces[0].root.exists()
    assert (repository / "module.py").read_text() == "VALUE = 1\n"
    saved = json.loads((evaluation.directory / "results.jsonl").read_text())
    assert saved["patch_status"] == "exported" and saved["success"] is True


async def test_patch_error_is_distinct_and_does_not_stop_later_tasks(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    repository = dataset / "repo"
    initialize_repository(repository)
    write_task(dataset, "first", "repo")
    write_task(dataset, "second", "repo")

    from codekeel.evals import runner

    real_export = runner.export_patch
    calls = 0

    def fail_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PatchExportError("API_KEY=secret")
        return real_export(*args, **kwargs)

    monkeypatch.setattr(runner, "export_patch", fail_first)

    def unchanged_model():
        return FakeModel([ModelResponse(content="done")])

    evaluation = await run_dataset(
        load_dataset(dataset), model_factory=unchanged_model, root=tmp_path / "results",
    )
    first, second = evaluation.results
    assert first.success and first.error is None and first.patch_status is PatchStatus.ERROR
    assert first.patch_path is first.patch_sha256 is first.patch_bytes is None
    assert second.success and second.patch_status is PatchStatus.EMPTY
    assert second.patch_path and Path(second.patch_path).is_file()
    assert "secret" not in (evaluation.directory / "results.jsonl").read_text()