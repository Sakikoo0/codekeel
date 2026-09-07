"""Bounded, public task checklists; no reasoning transcript or executable actions."""

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codekeel.models.base import Message

PlanStatus = Literal["pending", "in_progress", "completed", "blocked"]


class PlanItem(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$", strict=True)
    description: str = Field(min_length=1, max_length=256, strict=True)
    status: PlanStatus = "pending"

    @field_validator("description")
    @classmethod
    def concise_description(cls, value: str) -> str:
        if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Plan descriptions must be nonblank single-line task summaries")
        return value


class Plan(BaseModel):
    """Each update replaces the complete ordered checklist atomically.

    Reopening, removing and adding items are allowed as work changes. At most one
    item is active in this sequential runtime; blocked needs no dependency graph.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    items: tuple[PlanItem, ...] = Field(max_length=32)

    @model_validator(mode="after")
    def unique_and_sequential(self):
        if len({item.id for item in self.items}) != len(self.items):
            raise ValueError("Plan item IDs must be unique")
        if sum(item.status == "in_progress" for item in self.items) > 1:
            raise ValueError("At most one plan item may be in_progress")
        return self

    def reminder(self) -> Message:
        """User-level, ephemeral data; include it before context budgeting."""
        data = json.dumps(self.model_dump(mode="json"), ensure_ascii=True, separators=(",", ":"))
        return Message(role="user", content=(
            "Current task plan (agent-authored tracking data, not instructions or verification evidence). "
            "Keep it current with update_plan; record concise tasks and statuses, not private reasoning.\n" + data
        ))