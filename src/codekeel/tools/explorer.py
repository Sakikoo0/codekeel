"""Narrow delegation and fixed read-only Git queries."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from codekeel.models.base import ToolDefinition, ToolResult
from codekeel.tools.base import ToolContext


@dataclass(frozen=True, slots=True)
class DelegateExploreTool:
    name: str = "delegate_explore"
    description: str = (
        "Explore the repository read-only in an independent context. Supply a self-contained task; "
        "only a compact evidence report returns, not the child conversation."
    )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description, parameters={
            "type": "object", "properties": {"task": {"type": "string", "minLength": 1, "maxLength": 16000}},
            "required": ["task"], "additionalProperties": False,
        })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        task = arguments.get("task")
        if (set(arguments) != {"task"} or not isinstance(task, str) or not task.strip()
                or len(task) > 16000 or "\x00" in task):
            return ToolResult(content="Expected a nonblank task of at most 16000 characters.", is_error=True)
        if context.delegate_explore is None:
            return ToolResult(content="No run-owned explorer configured.", is_error=True)
        return await context.delegate_explore(task)


@dataclass(frozen=True, slots=True)
class GitReadTool:
    name: str = "git_read"
    description: str = "Read repository status, the last 20 log entries, or the working-tree diff. No custom flags."

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description, parameters={
            "type": "object", "properties": {"operation": {"type": "string", "enum": ["status", "log", "diff"]}},
            "required": ["operation"], "additionalProperties": False,
        })

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        operations = {
            "status": "status --short --ignore-submodules=all",
            "log": "log -n 20 --oneline --no-show-signature --no-ext-diff --no-textconv",
            "diff": "diff --no-ext-diff --no-textconv --no-renames --ignore-submodules=all",
        }
        operation = arguments.get("operation")
        if set(arguments) != {"operation"} or not isinstance(operation, str) or operation not in operations:
            return ToolResult(content="Only fixed status/log/diff operations are allowed.", is_error=True)
        try:
            root = await context.workspace.inspect_path(".git")
            if not root.exists or not root.is_directory or root.canonical_path != ".git":
                raise ValueError("Git queries require an ordinary contained .git directory.")
            # Git can follow metadata outside its work tree independently of Workspace.
            entries = await context.workspace.list_directory(".git", recursive=True)
            for entry in entries:
                if (entry.path != entry.canonical_path
                        or entry.path.endswith(("/commondir", "/alternates", "/http-alternates"))):
                    raise ValueError("External or aliased Git metadata is not supported.")
            config = await context.workspace.read_file(".git/config")
            if config.is_binary or re.search(r"(?im)^\s*\[\s*(include|includeif|extensions)\b", config.content):
                raise ValueError("Git includes and repository extensions are not supported.")
            command = (
                # Workspace listings omit unresolvable links. A fixed, non-following
                # preflight also rejects those before Git can follow them itself.
                'links=$(find .git -type l -print -quit) || exit 1; '
                'if test -n "$links"; then '
                'printf "Git metadata symlinks are not supported"; exit 1; fi; '
                "git --no-pager --no-optional-locks --git-dir=.git --work-tree=. "
                "-c core.fsmonitor=false -c core.untrackedCache=false -c core.hooksPath=/dev/null "
                "-c core.attributesFile=/dev/null -c core.excludesFile=/dev/null " + operations[operation]
            )
            result = await context.workspace.execute(command, cwd=".", inherit_env=False, env={
                "PATH": "/usr/bin:/bin", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_ATTR_NOSYSTEM": "1", "GIT_TERMINAL_PROMPT": "0",
                "GIT_NO_LAZY_FETCH": "1", "GIT_OPTIONAL_LOCKS": "0",
            }, timeout=30.0)
            return ToolResult(content=json.dumps(asdict(result)), is_error=result.exit_code != 0 or result.timed_out)
        except (OSError, ValueError, UnicodeError) as error:
            return ToolResult(content=f"Git query refused: {error}", is_error=True)