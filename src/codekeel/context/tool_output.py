"""Deterministic reduction of one tool result before it enters history."""

import hashlib
from dataclasses import dataclass

from codekeel.events.store import validate_run_id
from codekeel.models.base import ToolResult
from codekeel.workspace.base import Workspace

_LINE_ENDINGS = "\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029"


@dataclass(frozen=True, slots=True)
class ToolOutputConfig:
    """Limits apply to content, including the omission marker and retrieval hint.

    Spill thresholds count Unicode characters, not UTF-8 bytes or tokens.
    Reserve one line for the marker, and enough characters for a complete hint.
    """

    max_chars: int = 10_000
    max_lines: int = 256
    head_lines: int = 100
    tail_lines: int = 100
    spill_threshold: int = 100_000

    def __post_init__(self) -> None:
        for name in ("max_chars", "max_lines", "head_lines", "tail_lines", "spill_threshold"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_chars < 512:
            raise ValueError("max_chars must be at least 512 to retain the retrieval hint")
        if self.head_lines + self.tail_lines + 1 > self.max_lines:
            raise ValueError("max_lines must fit head_lines + tail_lines + one marker line")
        if self.spill_threshold <= self.max_chars:
            raise ValueError("spill_threshold must exceed max_chars")


class ToolOutputManager:
    def __init__(self, config: ToolOutputConfig | None = None) -> None:
        self.config = config or ToolOutputConfig()

    async def process(self, result: ToolResult, *, workspace: Workspace, run_id: str) -> ToolResult:
        """Return a bounded preview, preserving the tool's error flag.

        A failed spill yields a bounded, explicit failure notice, never a false
        retrieval hint or the original oversized text. Cancellation propagates.
        """
        validate_run_id(run_id)
        text = result.content
        lines = text.splitlines(keepends=True)
        if len(text) <= self.config.max_chars and len(lines) <= self.config.max_lines:
            return result

        note = "Full output not spilled."
        if len(text) >= self.config.spill_threshold:
            try:
                path = await self._spill(text, workspace, run_id)
                note = f"Full output; retrieve a slice with shell: sed -n '1,80p' {path}"
            except (OSError, ValueError, UnicodeError):
                note = "Full output spill failed; no artifact available."
        marker = f"[Output truncated: {len(text)} chars, {len(lines)} lines; middle omitted. {note}]"
        remaining = self.config.max_chars - len(marker) - 2
        # Both line and character limits apply, including a single huge line.
        head = "".join(lines[:self.config.head_lines])[:remaining // 2].rstrip(_LINE_ENDINGS)
        tail = "".join(lines[-self.config.tail_lines:])[-(remaining - remaining // 2):].lstrip(_LINE_ENDINGS)
        preview = f"{head}\n{marker}\n{tail}"
        return result.model_copy(update={"content": preview})

    async def _spill(self, text: str, workspace: Workspace, run_id: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        directory = f".agent/runs/{run_id}/artifacts"
        path = f"{directory}/{digest}.txt"
        # Reject aliases of every managed component, even aliases within the repo.
        for candidate in (".agent", ".agent/runs", f".agent/runs/{run_id}", directory):
            info = await workspace.inspect_path(candidate)
            if info.canonical_path != candidate or (info.exists and not info.is_directory):
                raise PermissionError("Unsafe artifact directory")
        info = await workspace.inspect_path(path)
        if info.canonical_path != path or info.is_directory:
            raise PermissionError("Unsafe artifact path")
        if info.exists:
            if info.size != len(text.encode("utf-8")):
                raise ValueError("Existing artifact differs")
            existing = await workspace.read_file(path)
            if existing.is_binary or existing.content != text:
                raise ValueError("Existing artifact differs")
        else:
            await workspace.write_file(path, text)
        return path