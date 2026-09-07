"""LiteLLM adapter for the provider-independent model contract."""

import json
from collections.abc import Mapping
from typing import Any

import litellm

from codekeel.models.base import Message, ModelResponse, ToolCall, ToolDefinition, Usage


class LiteLLMResponseError(ValueError):
    """Raised when LiteLLM returns a response that cannot be normalized."""

class LiteLLMModel:
    """Call any LiteLLM-supported provider through the project Model boundary."""

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("Model name cannot be blank")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be greater than zero")

        self.model = model
        self.api_base = api_base
        self.temperature = temperature
        self.max_tokens = max_tokens

    async def complete(
        self,
        messages: list[Message],
        *,
        tools: list[ToolDefinition] | None = None,
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [_message_to_provider(message) for message in messages],
        }
        if tools is not None:
            request["tools"] = [_tool_to_provider(tool) for tool in tools]
        if self.api_base is not None:
            request["api_base"] = self.api_base
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.max_tokens is not None:
            request["max_tokens"] = self.max_tokens

        response = await litellm.acompletion(**request)
        return _response_from_provider(response, model=self.model)


def _message_to_provider(message: Message) -> dict[str, Any]:
    provider_message: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_calls:
        provider_message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                },
            }
            for tool_call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        provider_message["tool_call_id"] = message.tool_call_id
    return provider_message


def _tool_to_provider(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _response_from_provider(response: Any, *, model: str) -> ModelResponse:
    choices = _get(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        raise LiteLLMResponseError("LiteLLM response did not include a choice")

    message = _get(choices[0], "message")
    content = _get(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise LiteLLMResponseError("LiteLLM response content must be a string or null")

    provider_tool_calls = _get(message, "tool_calls", None) or []
    if not isinstance(provider_tool_calls, (list, tuple)):
        raise LiteLLMResponseError("LiteLLM response tool_calls must be a list")

    usage = _get(response, "usage", None)
    return ModelResponse(
        content=content,
        tool_calls=[_tool_call_from_provider(tool_call) for tool_call in provider_tool_calls],
        usage=Usage(
            input_tokens=_token_count(usage, "prompt_tokens"),
            output_tokens=_token_count(usage, "completion_tokens"),
            cost=_response_cost(response, model=model),
        ),
    )


def _tool_call_from_provider(provider_tool_call: Any) -> ToolCall:
    function = _get(provider_tool_call, "function")
    arguments = _get(function, "arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as error:
            raise LiteLLMResponseError("LiteLLM tool call arguments are not valid JSON") from error
    if not isinstance(arguments, Mapping):
        raise LiteLLMResponseError("LiteLLM tool call arguments must be a JSON object")

    try:
        return ToolCall(
            id=_get(provider_tool_call, "id"),
            name=_get(function, "name"),
            arguments=dict(arguments),
        )
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise LiteLLMResponseError("LiteLLM returned an invalid tool call") from error


def _token_count(usage: Any, field: str) -> int:
    value = _get(usage, field, 0) if usage is not None else 0
    if value is None:
        return 0
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LiteLLMResponseError(f"LiteLLM usage {field} must be a non-negative integer")
    return value


def _response_cost(response: Any, *, model: str) -> float:
    hidden_params = _get(response, "_hidden_params", None)
    if isinstance(hidden_params, Mapping):
        value = hidden_params.get("response_cost")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
            return float(value)

    try:
        value = litellm.cost_calculator.completion_cost(response, model=model)
    except Exception:
        return 0.0
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        return 0.0
    return float(value)


def _get(value: Any, key: str, default: Any = ...) -> Any:
    if isinstance(value, Mapping):
        if default is ...:
            return value[key]
        return value.get(key, default)
    if default is ...:
        return getattr(value, key)
    return getattr(value, key, default)