"""Verification-grounded accounting, independent of model claims."""

from pydantic import BaseModel, ConfigDict, Field

from codekeel.agent.state import AgentState, RunStatus
from codekeel.evals.patch import PatchArtifact, PatchStatus
from codekeel.events.models import VerificationFinished
from codekeel.events.store import TraceReadResult


class EvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    task_id: str
    run_id: str
    success: bool
    status: RunStatus
    steps: int = Field(ge=0)
    model_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cost: float = Field(ge=0)
    duration: float = Field(ge=0)
    verification_result: bool | None
    verification_commands: int = Field(ge=0)
    trace_path: str | None
    error: str | None = None
    patch_status: PatchStatus = PatchStatus.ERROR
    patch_path: str | None = None
    patch_sha256: str | None = None
    patch_bytes: int | None = Field(default=None, ge=0)


def score(
    task_id: str, run_id: str, state: AgentState, trace: TraceReadResult,
    *, duration: float, trace_path: str | None, error: str | None = None,
    patch: PatchArtifact | None = None,
) -> EvaluationResult:
    """Only settled verification evidence can establish success.

    Missing/interrupted checks remain null; failed checks remain false. Ordinary
    tool calls exclude the separately reported runtime verification commands.
    """
    report = next((event for event in reversed(trace.events) if isinstance(event, VerificationFinished)), None)
    verified = report.payload.passed if report is not None and trace.warning is None else None
    patch = patch or PatchArtifact.error()
    return EvaluationResult(
        task_id=task_id, run_id=run_id, status=state.status,
        success=(verified is True and state.verification_passed is True
                 and state.status is RunStatus.COMPLETED and error is None),
        steps=state.steps, model_calls=state.model_calls, tool_calls=state.tool_calls,
        input_tokens=state.usage.input_tokens, output_tokens=state.usage.output_tokens, cost=state.usage.cost,
        duration=duration, verification_result=verified, verification_commands=state.verification_commands,
        trace_path=trace_path, error=error, patch_status=patch.status, patch_path=patch.path,
        patch_sha256=patch.sha256, patch_bytes=patch.size,
    )