"""Validated resource limits for one agent run."""

from pydantic import BaseModel, ConfigDict, Field


class BudgetLimits(BaseModel):
    """Maximum work and model usage allowed during one run.

    ``None`` disables an individual limit. The finite default step and wall-time
    limits ensure the default agent cannot loop forever.
    """

    model_config = ConfigDict(frozen=True, strict=True, allow_inf_nan=False)

    max_steps: int | None = Field(default=500, gt=0)
    max_model_calls: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    max_wall_time: float | None = Field(default=3600.0, gt=0)
    max_input_tokens: int | None = Field(default=None, gt=0)
    max_output_tokens: int | None = Field(default=None, gt=0)
    max_cost: float | None = Field(default=None, gt=0)