"""Deterministic model implementations for tests."""

from collections import deque
from collections.abc import Iterable

from codekeel.models.base import Message, ModelResponse, ToolDefinition


class ScriptedModel:
    """Return predefined model responses in order."""

    def __init__(self, responses: Iterable[ModelResponse]) -> None:
        self._responses = deque(responses)

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        if not self._responses:
            raise RuntimeError("Scripted model has no responses remaining")
        return self._responses.popleft()

class FakeModel(ScriptedModel):
    """Test-friendly name for a scripted deterministic model."""