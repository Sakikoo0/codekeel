import pytest
from pydantic import ValidationError

from codekeel.models import Message, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage


def test_model_contracts_round_trip() -> None:
    tool = ToolDefinition(
        name="shell",
        description="Run a command",
        parameters={"type": "object", "properties": {"command": {"type": "string"}}},
    )
    call = ToolCall(id="call-1", name=tool.name, arguments={"command": "pwd"})
    response = ModelResponse(
        content=None,
        tool_calls=[call],
        usage=Usage(input_tokens=10, output_tokens=4, cost=0.01),
    )
    result = ToolResult(content="/workspace")
    message = Message(role="tool", content=result.content, tool_call_id=call.id)

    assert ModelResponse.model_validate_json(response.model_dump_json()) == response
    assert message.model_dump(mode="json") == {
        "role": "tool",
        "content": "/workspace",
        "tool_calls": [],
        "tool_call_id": "call-1",
    }


def test_tool_definition_rejects_blank_name() -> None:
    with pytest.raises(ValidationError, match="Tool name cannot be blank"):
        ToolDefinition(name="  ", description="invalid")


def test_usage_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        Usage(input_tokens=-1)
