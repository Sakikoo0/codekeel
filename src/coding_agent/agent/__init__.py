"""Agent runtime and state contracts."""

from typing import TYPE_CHECKING

from coding_agent.agent.state import AgentState, RunStatus
from coding_agent.agent.termination import TerminationPolicy

if TYPE_CHECKING:
    from coding_agent.agent.loop import Agent, AgentProtocolError, FinalAnswer

__all__ = ["Agent", "AgentProtocolError", "AgentState", "FinalAnswer", "RunStatus", "TerminationPolicy"]


def __getattr__(name: str):
    if name in {"Agent", "AgentProtocolError", "FinalAnswer"}:
        from coding_agent.agent import loop
        return getattr(loop, name)
    raise AttributeError(name)