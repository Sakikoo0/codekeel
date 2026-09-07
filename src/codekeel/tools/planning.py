"""Whole-plan replacement through an injected, run-owned update callback."""

from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from codekeel.models.base import ToolDefinition, ToolResult
from codekeel.planning import Plan
from codekeel.tools.base import ToolContext


@dataclass(frozen=True, slots=True)
class UpdatePlanTool:
    name: str = "update_plan"
    description: str = (
        "Create or replace the whole ordered task plan for multi-step work. Pass items with stable id, "
        "concise description and status (pending, in_progress, completed, blocked), including unchanged items. "
        "Keep at most one item in_progress. Mark completed only when done; use blocked for unresolved obstacles. "
        "Use an empty items list to clear the plan. Record public task summaries, never private reasoning."
    )

    def definition(self) -> ToolDefinition:
        return ToolDefinition(name=self.name, description=self.description, parameters=Plan.model_json_schema())

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        try:
            plan = Plan.model_validate(arguments)
        except ValidationError as error:
            # Do not echo untrusted extra fields (such as reasoning) or oversized input.
            reasons = sorted({entry["msg"] for entry in error.errors(include_input=False)})
            return ToolResult(content="Plan not updated: " + "; ".join(reasons), is_error=True)
        if context.update_plan is None:
            return ToolResult(content="Plan not updated: no run-owned planning callback configured.", is_error=True)
        context.update_plan(plan)
        return ToolResult(content=f"Plan updated: {len(plan.items)} item(s).")