"""Serializable action guardrails; execution isolation remains the workspace's job."""

import re
import shlex
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from codekeel.models.base import ToolCall


class Decision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class Risk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class CommandRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prefix: tuple[str, ...] = Field(min_length=1)
    risk: Risk
    deny: bool = False

    @field_validator("prefix")
    @classmethod
    def nonempty_words(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not word or word.strip() != word for word in value):
            raise ValueError("Command rule words must not be empty or padded")
        return value


def _command_rules() -> tuple[CommandRule, ...]:
    return tuple(CommandRule(prefix=prefix, risk=risk, deny=deny) for prefix, risk, deny in [
        (("git", "push"), Risk.HIGH, True),
        (("git", "clean"), Risk.HIGH, True),
        (("git", "reset"), Risk.HIGH, True),
        *[((name,), Risk.HIGH, True) for name in (
            "rm", "rmdir", "mkfs", "dd", "format", "shutdown", "reboot", "halt", "poweroff", "init",
        )],
        (("pip", "install"), Risk.HIGH, False),
        (("pip3", "install"), Risk.HIGH, False),
        (("python", "-m", "pip", "install"), Risk.HIGH, False),
        (("python3", "-m", "pip", "install"), Risk.HIGH, False),
        (("git", "commit"), Risk.HIGH, False),
        *[((name,), Risk.LOW, False) for name in ("pytest", "echo", "printf", "pwd", "ls", "cat")],
        (("git", "status"), Risk.LOW, False),
        (("git", "diff"), Risk.LOW, False),
        (("git", "log", "--oneline", "-5", "--"), Risk.LOW, False),
    ])


class Assessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    risk: Risk
    decision: Decision
    reason: str | None = None


class ActionPolicy(BaseModel):
    """Deny wins in every mode; unknown actions require review by default.

    Rules match literal argv prefixes, never model-supplied risk labels. Compound
    shell syntax is refused, not parsed as a program. These are accident guardrails:
    even pytest can execute arbitrary repository code.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: Literal["always", "never", "risky"] = "risky"
    approval_handling: Literal["pause", "unavailable"] = "pause"
    threshold: Risk = Risk.HIGH
    confirm_unknown: bool = True
    tool_risks: dict[str, Risk] = Field(default_factory=lambda: {
        **dict.fromkeys(("read_file", "list_directory", "find_files", "search_files"), Risk.LOW),
        "delegate_explore": Risk.LOW, "git_read": Risk.LOW,
        "write_file": Risk.MEDIUM, "edit_file": Risk.MEDIUM, "update_plan": Risk.LOW,
    })
    denied_tools: tuple[str, ...] = ()
    command_rules: tuple[CommandRule, ...] = Field(default_factory=_command_rules)

    @field_validator("threshold")
    @classmethod
    def known_threshold(cls, value: Risk) -> Risk:
        if value is Risk.UNKNOWN:
            raise ValueError("Approval threshold cannot be UNKNOWN")
        return value

    def assess(self, call: ToolCall) -> Assessment:
        risk = self.tool_risks.get(call.name, Risk.UNKNOWN)
        denied = call.name in self.denied_tools
        reason = "This tool is explicitly denied by policy." if denied else None
        if call.name == "shell":
            risk, shell_denied, shell_reason = self._shell(call.arguments.get("command"))
            denied |= shell_denied
            reason = reason or shell_reason
        if denied:
            return Assessment(risk=risk, decision=Decision.DENY, reason=reason)
        levels = [Risk.LOW, Risk.MEDIUM, Risk.HIGH]
        confirm = self.confirm_unknown if risk is Risk.UNKNOWN else levels.index(risk) >= levels.index(self.threshold)
        if self.mode == "always" or (self.mode == "risky" and confirm):
            return Assessment(risk=risk, decision=Decision.REQUIRE_APPROVAL)
        return Assessment(risk=risk, decision=Decision.ALLOW)

    def _shell(self, command: object) -> tuple[Risk, bool, str | None]:
        if not isinstance(command, str) or not command.strip():
            return Risk.UNKNOWN, False, None  # Tool validation reports malformed arguments.
        # Fixed messages do not echo potentially sensitive command contents.
        if "<<" in command or "\n" in command or "\r" in command:
            return Risk.UNKNOWN, True, (
                "Heredocs and multiline commands are not supported by shell policy. "
                "If write_file is available, save the script with it, then invoke the script using an allowed "
                "single-line command. Saving a script does not grant permission to execute it."
            )
        # Refuse expansions, operators, redirections, comments and line continuations,
        # even in quotes. Deliberately conservative rather than a partial shell parser.
        if re.search(r"[;&|<>$`(){}\[\]*?~#!\\\x00-\x1f\x7f]", command):
            return Risk.UNKNOWN, True, (
                "Shell policy rejects operators, redirections, expansions, control characters and special "
                "characters, even inside quotes. Use one simple command without these characters; "
                "use read_file, search_files, write_file or edit_file when available for file operations."
            )
        try:
            words = tuple(shlex.split(command))
        except ValueError:
            return Risk.UNKNOWN, True, "Command quoting is invalid. Use balanced quotes in one simple command."
        if not words:
            return Risk.UNKNOWN, True, "No executable was found. Supply one simple command."
        matches = [rule for rule in self.command_rules if words[:len(rule.prefix)] == rule.prefix]
        # Denied executables cannot be disguised with an absolute/relative path.
        normalized = (words[0].rsplit("/", 1)[-1], *words[1:])
        for rule in self.command_rules:
            if rule.deny and (normalized[:len(rule.prefix)] == rule.prefix or (
                len(rule.prefix) == 2 and normalized[0] == rule.prefix[0] and rule.prefix[1] in normalized[1:]
            )):
                return rule.risk, True, "Command explicitly denied by policy. Choose an allowed tool or command."
        if not matches:
            return Risk.UNKNOWN, False, None
        rule = max(matches, key=lambda item: len(item.prefix))
        return rule.risk, rule.deny, None