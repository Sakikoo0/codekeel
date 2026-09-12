from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from codekeel.models import (
    LiteLLMModel,
    LiteLLMResponseError,
    Message,
    ToolCall,
    ToolDefinition,
    Usage,
)


def _provider_response(
    *,
    content: str | None,
    tool_calls: list[object] | None = None,
    input_tokens: int = 11,
    output_tokens: int = 7,
    cost: float = 0.004,
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=SimpleNamespace(prompt_tokens=input_tokens, completion_tokens=output_tokens),
        _hidden_params={"response_cost": cost},
    )


async def test_litellm_model_converts_text_response_and_forwards_config(monkeypatch):
    completion = AsyncMock(return_value=_provider_response(content="done"))
    monkeypatch.setattr("codekeel.models.litellm.litellm.acompletion", completion)
    model = LiteLLMModel(
        "openai/test-model",
        api_base="https://models.example/v1",
        temperature=0.25,
        max_tokens=512,
    )
    tools = [
        ToolDefinition(
            name="shell",
            description="Run a command",
            parameters={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        )
    ]

    response = await model.complete([Message(role="user", content="finish")], tools=tools)

    assert response.content == "done"
    assert response.tool_calls == []
    assert response.usage == Usage(input_tokens=11, output_tokens=7, cost=0.004)
    completion.assert_awaited_once_with(
        model="openai/test-model",
        api_base="https://models.example/v1",
        temperature=0.25,
        max_tokens=512,
        messages=[{"role": "user", "content": "finish"}],
        parallel_tool_calls=False,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "shell",
                    "description": "Run a command",
                    "parameters": tools[0].parameters,
                },
            }
        ],
    )


async def test_litellm_model_round_trips_tool_messages_and_calls(monkeypatch):
    provider_tool_call = SimpleNamespace(
        id="call-2",
        function=SimpleNamespace(name="shell", arguments='{"command":"pwd"}'),
    )
    completion = AsyncMock(
        return_value=_provider_response(content=None, tool_calls=[provider_tool_call])
    )
    monkeypatch.setattr("codekeel.models.litellm.litellm.acompletion", completion)
    model = LiteLLMModel("test/model")

    response = await model.complete(
        [
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="call-1", name="shell", arguments={"command": "ls"})
                ],
            ),
            Message(role="tool", content="files", tool_call_id="call-1"),
        ]
    )

    assert response.tool_calls == [
        ToolCall(id="call-2", name="shell", arguments={"command": "pwd"})
    ]
    assert completion.await_args.kwargs["messages"] == [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "shell", "arguments": '{"command": "ls"}'},
                }
            ],
        },
        {"role": "tool", "content": "files", "tool_call_id": "call-1"},
    ]
    assert "parallel_tool_calls" not in completion.await_args.kwargs


async def test_litellm_model_propagates_provider_failure(monkeypatch):
    completion = AsyncMock(side_effect=RuntimeError("provider unavailable"))
    monkeypatch.setattr("codekeel.models.litellm.litellm.acompletion", completion)

    with pytest.raises(RuntimeError, match="provider unavailable"):
        await LiteLLMModel("test/model").complete([Message(role="user", content="hello")])


async def test_litellm_model_rejects_malformed_tool_arguments(monkeypatch):
    provider_tool_call = SimpleNamespace(
        id="call-1",
        function=SimpleNamespace(name="shell", arguments="not-json"),
    )
    completion = AsyncMock(
        return_value=_provider_response(content=None, tool_calls=[provider_tool_call])
    )
    monkeypatch.setattr("codekeel.models.litellm.litellm.acompletion", completion)

    with pytest.raises(LiteLLMResponseError, match="not valid JSON"):
        await LiteLLMModel("test/model").complete([Message(role="user", content="hello")])


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_litellm_model_rejects_non_positive_max_tokens(max_tokens):
    with pytest.raises(ValueError, match="greater than zero"):
        LiteLLMModel("test/model", max_tokens=max_tokens)
