"""Provider-independent model and message contracts."""

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, field_validator


class ToolDefinition(BaseModel):
    """JSON-schema description of a tool available to a model."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Tool name cannot be blank")
        return value

class ToolCall(BaseModel):
    """A model request to invoke one tool."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)

class ToolResult(BaseModel):
    """Normalized result returned by a tool invocation."""

    content: str
    is_error: bool = False

class Message(BaseModel):
    """A provider-independent conversation message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

class Usage(BaseModel):
    """Token and monetary usage reported for one or more model calls."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost: float = Field(default=0.0, ge=0.0)

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost=self.cost + other.cost
        )

class ModelResponse(BaseModel):
    """Normalized response returned by a model provider."""

    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)

class Model(Protocol):
    """Boundary implemented by model providers and deterministic test models."""

    async def complete(
            self,
            messages: list[Message],
            *,
            tools: list[ToolDefinition] | None = None
    ) -> ModelResponse: ...