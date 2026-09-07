"""Explicit, independent feature selections for evaluation composition."""

import stat
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from codekeel.agent import Agent
from codekeel.agent.explorer import ExplorerAgent
from codekeel.context.compaction import DeterministicContextManager
from codekeel.context.tool_output import ToolOutputManager
from codekeel.models.base import Message, Model, ToolDefinition, ToolResult
from codekeel.tools.registry import ToolRegistry
from codekeel.workspace.base import Workspace


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    repo_context: bool = False
    tool_output_limits: bool = False
    context_compaction: bool = False
    planning: bool = False
    explorer: bool = False


class _ConfigLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        # Do not silently override a toggle through duplicate keys or YAML merges.
        keys = [self.construct_object(key, deep=deep) for key, _ in node.value]
        if any(not isinstance(key, str) for key in keys) or len(set(keys)) != len(keys):
            raise ValueError("Config keys must be unique strings")
        return super().construct_mapping(node, deep=deep)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 64_000:
        raise ValueError("Config must be a regular YAML file of at most 64 KB")
    return ExperimentConfig.model_validate(yaml.load(path.read_text(encoding="utf-8"), Loader=_ConfigLoader))


class UnchangedContext:
    """Independent copies without history reduction for compaction ablations."""

    def prepare(self, messages: list[Message], *, tools: list[ToolDefinition] | None = None) -> list[Message]:
        return [message.model_copy(deep=True) for message in messages]


class UnchangedToolOutput(ToolOutputManager):
    """Disable only the pre-history reducer; tool/workspace bounds still apply."""

    async def process(self, result: ToolResult, *, workspace: Workspace, run_id: str) -> ToolResult:
        return result.model_copy(deep=True)


def configure_agent(
    agent: Agent, config: ExperimentConfig, *, model_factory: Callable[[], Model],
) -> None:
    """Compose a newly constructed, not-yet-started evaluation agent only."""
    if agent.run_id is not None:
        raise ValueError("Experiment configuration requires a fresh agent")
    registry = agent.tool_registry
    agent.tool_registry = ToolRegistry(
        registry.get(item.name) for item in registry.definitions()
        if item.name not in {"update_plan", "delegate_explore"}
        or (item.name == "update_plan" and config.planning)
    )
    agent.context_manager = DeterministicContextManager() if config.context_compaction else UnchangedContext()
    agent.tool_output_manager = ToolOutputManager() if config.tool_output_limits else UnchangedToolOutput()
    agent.explorer = None
    if config.explorer:
        from codekeel.tools.explorer import DelegateExploreTool

        agent.explorer = ExplorerAgent(model_factory())
        agent.tool_registry.register(DelegateExploreTool())
