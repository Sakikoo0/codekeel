"""Agent runtime and state contracts."""

from typing import TYPE_CHECKING

from codekeel.agent.state import AgentState, RunStatus
from codekeel.agent.termination import TerminationPolicy

if TYPE_CHECKING:
    from codekeel.agent.loop import Agent, AgentProtocolError, FinalAnswer

__all__ = ["Agent", "AgentProtocolError", "AgentState", "FinalAnswer", "RunStatus", "TerminationPolicy"]


def __getattr__(name: str):
    if name in {"Agent", "AgentProtocolError", "FinalAnswer"}:
        from codekeel.agent import loop
        return getattr(loop, name)
    raise AttributeError(name)