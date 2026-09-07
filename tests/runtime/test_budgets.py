import pytest
from pydantic import ValidationError

from codekeel.runtime import BudgetLimits


def test_budget_defaults_provide_hard_run_bounds() -> None:
    budgets = BudgetLimits()

    assert budgets.max_steps == 500
    assert budgets.max_wall_time == 3600.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_model_calls", -1),
        ("max_tool_calls", True),
        ("max_wall_time", float("inf")),
        ("max_input_tokens", "10"),
        ("max_output_tokens", 0),
        ("max_cost", -0.01),
    ],
)
def test_budget_limits_reject_invalid_values(field, value) -> None:
    with pytest.raises(ValidationError):
        BudgetLimits(**{field: value})


def test_individual_limits_can_be_disabled() -> None:
    budgets = BudgetLimits(max_steps=None, max_wall_time=None)

    assert budgets.max_steps is None
    assert budgets.max_wall_time is None