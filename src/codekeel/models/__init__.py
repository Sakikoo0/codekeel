"""Model contracts and deterministic implementations."""

from typing import TYPE_CHECKING

from codekeel.models.base import (
    Message,
    Model,
    ModelResponse,
    ToolCall,
    ToolDefinition,
    ToolResult,
    Usage,
)
from codekeel.models.fake import FakeModel, ScriptedModel

if TYPE_CHECKING:
    from codekeel.models.litellm import LiteLLMModel, LiteLLMResponseError

__all__ = [
    "FakeModel",
    "LiteLLMModel",
    "LiteLLMResponseError",
    "Message",
    "Model",
    "ModelResponse",
    "ScriptedModel",
    "ToolCall",
    "ToolDefinition",
    "ToolResult",
    "Usage",
]

def __getattr__(name: str):
    """Keep contract-only clients (including trace inspection) provider independent."""
    if name in {"LiteLLMModel", "LiteLLMResponseError"}:
        from codekeel.models.litellm import LiteLLMModel, LiteLLMResponseError

        return {"LiteLLMModel": LiteLLMModel, "LiteLLMResponseError": LiteLLMResponseError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")