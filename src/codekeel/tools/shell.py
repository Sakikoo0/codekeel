"""Bounded shell tool policies executed through a workspace."""

import fnmatch
import math
import os
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from codekeel.models.base import ToolDefinition, ToolResult
from codekeel.tools.base import ToolArgumentsError, ToolContext
from codekeel.workspace.models import CommandResult

DEFAULT_DENIED_COMMANDS: tuple[str, ...] = (
    "rm",
    "rmdir",
    "mkfs",
    "dd",
    "format",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
)

DEFAULT_ENV_DENY_PATTERNS: tuple[str, ...] = (
    "ANTHROPIC_*",
    "AWS_*",
    "AZURE_*",
    "GATEWAY_*",
    "GEMINI_*",
    "GITHUB_TOKEN",
    "GITLAB_TOKEN",
    "GOOGLE_*",
    "HF_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
    "NPM_TOKEN",
    "OPENAI_*",
    "OPENROUTER_*",
    "PYDANTIC_AI_GATEWAY_API_KEY",
    "PYPI_TOKEN",
)

_RECOVERABLE_ERRORS = (FileNotFoundError, NotADirectoryError, PermissionError, ValueError)


@dataclass(frozen=True, slots=True)
class ShellConfig:
    """Policy and resource limits for one workspace-rooted shell tool."""

    allowed_commands: tuple[str, ...] = ()
    denied_commands: tuple[str, ...] | None = None
    deny_patterns: tuple[str, ...] = ()
    timeout: float = 30.0
    max_output_bytes: int = 50_000
    env_allowlist: tuple[str, ...] = ()
    env_deny_patterns: tuple[str, ...] = DEFAULT_ENV_DENY_PATTERNS
    cwd: str | Path = "."

    def __post_init__(self) -> None:
        allowed_commands = _string_tuple("allowed_commands", self.allowed_commands)
        denied_commands = (
            (() if allowed_commands else DEFAULT_DENIED_COMMANDS)
            if self.denied_commands is None
            else _string_tuple("denied_commands", self.denied_commands)
        )
        if allowed_commands and denied_commands:
            raise ValueError("Specify allowed_commands or denied_commands, not both")

        deny_patterns = _string_tuple("deny_patterns", self.deny_patterns)
        for pattern in deny_patterns:
            try:
                re.compile(pattern)
            except re.error as error:
                raise ValueError(f"Invalid deny pattern {pattern!r}: {error}") from error

        env_allowlist = _string_tuple("env_allowlist", self.env_allowlist)
        env_deny_patterns = _string_tuple("env_deny_patterns", self.env_deny_patterns)
        if (
            not isinstance(self.timeout, int | float)
            or isinstance(self.timeout, bool)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        if not isinstance(self.max_output_bytes, int) or isinstance(self.max_output_bytes, bool):
            raise ValueError("max_output_bytes must be a positive integer")
        if self.max_output_bytes <= 0:
            raise ValueError("max_output_bytes must be a positive integer")

        if not isinstance(self.cwd, str | Path):
            raise ValueError("cwd must be a string or Path")
        if isinstance(self.cwd, str) and not self.cwd.strip():
            raise ValueError("cwd cannot be blank")
        cwd = Path(self.cwd)
        if cwd.is_absolute():
            raise ValueError("cwd must be relative to the workspace root")
        if ".." in cwd.parts:
            raise ValueError("cwd cannot contain '..'")
        object.__setattr__(self, "allowed_commands", allowed_commands)
        object.__setattr__(self, "denied_commands", denied_commands)
        object.__setattr__(self, "deny_patterns", deny_patterns)
        object.__setattr__(self, "env_allowlist", env_allowlist)
        object.__setattr__(self, "env_deny_patterns", env_deny_patterns)
        object.__setattr__(self, "cwd", cwd.as_posix())

    def validate_command(self, command: str) -> None:
        """Apply best-effort command guardrails before workspace execution."""
        if "\x00" in command:
            raise PermissionError("Command contains a NUL byte")
        try:
            os.fsencode(command)
        except UnicodeEncodeError as error:
            raise PermissionError("Command cannot be encoded for the operating system") from error

        for pattern in self.deny_patterns:
            if re.search(pattern, command):
                raise PermissionError(f"Command is denied by pattern {pattern!r}")

        try:
            tokens = shlex.split(command)
        except ValueError as error:
            raise PermissionError(f"Command could not be parsed: {error}") from error
        executable = tokens[0]
        if self.denied_commands and executable in self.denied_commands:
            raise PermissionError(f"Command {executable!r} is denied")
        if self.allowed_commands and executable not in self.allowed_commands:
            raise PermissionError(f"Command {executable!r} is not in the allowed list")

    def filtered_environment(self, environment: Mapping[str, str]) -> dict[str, str]:
        """Filter inherited variables before the process is spawned."""
        return {
            name: value
            for name, value in environment.items()
            if (not self.env_allowlist or name in self.env_allowlist)
            and not any(fnmatch.fnmatchcase(name, pattern) for pattern in self.env_deny_patterns)
        }


@dataclass(frozen=True, slots=True)
class ShellTool:
    """Run one bounded command through the configured workspace."""

    config: ShellConfig = field(default_factory=ShellConfig)
    name: str = "shell"
    description: str = (
        "Run a bounded command in the workspace. Default agent policy requires one simple single-line command: "
        "no heredocs, pipes, redirections, command chaining, expansions or special characters, even inside quotes. "
        "Use file tools when available to read, search, write or edit files. For scripts, write the file first; "
        "execution still requires an allowed command. Default policy permits bounded history lookup as "
        "git log --oneline -5 -- path/to/file; other git log forms may require approval. "
        "Commands run from the configured workspace directory; do not assume /workspace exists. "
        "This policy is a guardrail, not a security boundary."
    )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
                "additionalProperties": False,
            },
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> None:
        """Validate the call shape without any workspace access."""
        _validate_arguments(self.name, arguments)
        _non_empty_string(self.name, "command", arguments["command"])

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        self.validate_arguments(arguments)
        command = arguments["command"]
        try:
            self.config.validate_command(command)
            result = await context.workspace.execute(
                command,
                cwd=self.config.cwd,
                env=self.config.filtered_environment(os.environ),
                timeout=self.config.timeout,
                inherit_env=False,
            )
        except _RECOVERABLE_ERRORS as error:
            return ToolResult(
                content=_truncate_tail(str(error), self.config.max_output_bytes),
                is_error=True,
            )

        content = _format_result(result, self.config.timeout)
        if not context.defer_output_limits:
            content = _truncate_tail(content, self.config.max_output_bytes)
        return ToolResult(content=content, is_error=result.exit_code != 0 or result.timed_out)


def _format_result(result: CommandResult, timeout: float) -> str:
    sections: list[str] = []
    if result.stdout:
        sections.append(f"[stdout]\n{result.stdout}")
    if result.stderr:
        sections.append(f"[stderr]\n{result.stderr}")
    if not sections:
        sections.append("(no output)")
    if result.timed_out:
        sections.append(f"[timed out after {timeout:g}s]")
    sections.append(f"[exit code: {result.exit_code}]")
    return "\n".join(sections)


def _truncate_tail(content: str, maximum: int) -> str:
    encoded = content.encode("utf-8")
    if len(encoded) <= maximum:
        return content
    marker = b"[... output truncated]\n"
    if maximum <= len(marker):
        return marker[:maximum].decode("ascii")
    tail = encoded[-(maximum - len(marker)) :]
    while tail and tail[0] & 0b1100_0000 == 0b1000_0000:
        tail = tail[1:]
    return marker.decode("ascii") + tail.decode("utf-8", errors="ignore")


def _validate_arguments(tool_name: str, arguments: dict[str, Any]) -> None:
    supplied = set(arguments)
    if "command" not in supplied:
        raise ToolArgumentsError(f"Tool {tool_name!r} is missing required argument(s): command")
    if unexpected := supplied - {"command"}:
        raise ToolArgumentsError(
            f"Tool {tool_name!r} received unexpected argument(s): {', '.join(sorted(unexpected))}"
        )


def _non_empty_string(tool_name: str, argument_name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} must be a string")
    if not value.strip():
        raise ToolArgumentsError(f"Tool {tool_name!r} argument {argument_name!r} cannot be blank")
    return value


def _string_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence of non-empty strings")
    normalized = tuple(values)
    if any(not isinstance(value, str) or not value.strip() for value in normalized):
        raise ValueError(f"{name} must contain non-empty strings")
    return normalized