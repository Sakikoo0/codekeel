"""Deterministic termination decisions for agent runs."""

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from coding_agent.agent.state import AgentState, RunStatus
from coding_agent.runtime.budgets import BudgetLimits


@dataclass(frozen=True, slots=True)
class TerminationPolicy:
    """Map accumulated work and elapsed time to a terminal run status."""

    budgets: BudgetLimits = field(default_factory=BudgetLimits)
    clock: Callable[[], float] = time.monotonic

    def evaluate(self, state: AgentState, *, started_at: float) -> RunStatus | None:
        """Return the first exceeded limit in deterministic priority order."""
        if self.wall_time_exceeded(started_at=started_at):
            return RunStatus.TIMEOUT
        if _reached(state.usage.cost, self.budgets.max_cost):
            return RunStatus.MAX_COST
        if _reached(state.usage.input_tokens, self.budgets.max_input_tokens):
            return RunStatus.MAX_TOKENS
        if _reached(state.usage.output_tokens, self.budgets.max_output_tokens):
            return RunStatus.MAX_TOKENS
        if _reached(state.steps, self.budgets.max_steps):
            return RunStatus.MAX_STEPS
        if _reached(state.model_calls, self.budgets.max_model_calls):
            return RunStatus.MAX_STEPS
        if _reached(state.tool_calls, self.budgets.max_tool_calls):
            return RunStatus.MAX_STEPS
        return None

    def wall_time_exceeded(self, *, started_at: float) -> bool:
        """Return whether the wall-clock deadline has been reached."""
        remaining = self.remaining_wall_time(started_at=started_at)
        return remaining is not None and remaining <= 0

    def remaining_wall_time(self, *, started_at: float) -> float | None:
        """Return seconds remaining, or ``None`` when wall time is unlimited."""
        if self.budgets.max_wall_time is None:
            return None
        return self.budgets.max_wall_time - max(0.0, self.clock() - started_at)


def _reached(value: int | float, limit: int | float | None) -> bool:
    return limit is not None and value >= limit