from codekeel.agent import AgentState, RunStatus
from codekeel.models import Usage


def test_agent_state_usage_accumulates() -> None:
    state = AgentState()

    state.add_usage(Usage(input_tokens=10, output_tokens=4, cost=0.02))
    state.add_usage(Usage(input_tokens=3, output_tokens=2, cost=0.01))

    assert state.usage == Usage(input_tokens=13, output_tokens=6, cost=0.03)


def test_run_status_serializes() -> None:
    state = AgentState(status=RunStatus.COMPLETED)

    assert state.model_dump(mode="json")["status"] == "completed"
    assert AgentState.model_validate_json(state.model_dump_json()).status is RunStatus.COMPLETED
