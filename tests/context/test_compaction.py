import json
from unittest.mock import AsyncMock

import pytest

from codekeel.agent import Agent, RunStatus
from codekeel.context.compaction import ContextConfig, DeterministicContextManager, estimate_context_tokens
from codekeel.context.manager import ContextBudgetExceeded, ContextHistoryError
from codekeel.models import FakeModel, Message, ModelResponse, ToolCall, ToolDefinition, ToolResult, Usage
from codekeel.runtime.policy import ActionPolicy, Risk
from codekeel.tools import ToolRegistry


def prefix():
    return [Message(role="system", content="rules"), Message(role="user", content="initial task")]


def pair(index, content="output", *, arguments=None):
    return [
        Message(role="assistant", content=f"turn {index}", tool_calls=[
            ToolCall(id=str(index), name="record", arguments=arguments or {}),
        ]),
        Message(role="tool", tool_call_id=str(index),
                content=ToolResult(content=content, is_error=True).model_dump_json()),
    ]


def manager(**changes):
    return DeterministicContextManager(ContextConfig(**changes))


def assert_pairs(messages):
    pending = set()
    for message in messages:
        if message.role == "tool":
            assert message.tool_call_id in pending
            pending.remove(message.tool_call_id)
        else:
            assert not pending
            pending.update(call.id for call in message.tool_calls)
    assert not pending


def test_120_turn_window_preserves_task_system_and_recent_turns():
    history = prefix() + [message for index in range(120) for message in pair(index, "x" * 200)]
    original = [message.model_copy(deep=True) for message in history]
    compact = manager(max_tokens=1000, keep_recent_turns=3)
    result = compact.prepare(history)
    assert estimate_context_tokens(result) <= 1000
    assert result[:2] == history[:2]
    assert result[-6:] == history[-6:]
    assert len(result) < len(history)
    assert history == original
    assert_pairs(result)
    assert compact.prepare(result) == result
    result[0].content = "modified"
    assert history[0].content == "rules"


def test_clear_old_results_before_dropping_turns_and_preserve_error_flag():
    history = prefix() + pair(0, "x" * 3000) + pair(1)
    result = manager(max_tokens=400, keep_recent_turns=1).prepare(history)
    assert len(result) == len(history)
    assert json.loads(result[3].content) == {"content": "[tool result cleared]", "is_error": True}
    assert result[-2:] == history[-2:]
    assert result[2] == history[2]


def test_small_history_unchanged_and_tool_definitions_counted():
    history = prefix() + pair(0)
    assert manager().prepare(history) == history
    tools = [ToolDefinition(name="record", description="界" * 3000, parameters={"type": "object"})]
    assert estimate_context_tokens(history, tools) > estimate_context_tokens(history)
    with pytest.raises(ContextBudgetExceeded):
        manager(max_tokens=300).prepare(history, tools=tools)


def test_plain_result_clearing_does_not_expand_small_results():
    history = prefix() + pair(0) + pair(1) + pair(2)
    history[3].content = "ok"
    history[5].content = "plain result " * 1000
    expected = [message.model_copy(deep=True) for message in history]
    expected[5].content = "[tool result cleared]"
    result = manager(max_tokens=estimate_context_tokens(expected), keep_recent_turns=1).prepare(history)
    assert result == expected


def test_recent_tool_output_that_cannot_fit_fails_without_breaking_pair():
    history = prefix() + pair(0, "x" * 10_000)
    with pytest.raises(ContextBudgetExceeded):
        manager(max_tokens=100, keep_recent_turns=1).prepare(history)
    assert_pairs(history)
    assert json.loads(history[-1].content)["content"] == "x" * 10_000


@pytest.mark.parametrize("text", ["H" + " " * 1_500_000 + "T", "界" * 5000, "\x00" * 5000])
@pytest.mark.parametrize("limit", [64, 128])
def test_pathological_response_and_arguments_clamp_is_idempotent(text, limit):
    history = prefix() + pair(0, arguments={"text": text})
    history[2].content = text
    compact = manager(max_tokens=2000, max_message_chars=limit, keep_recent_turns=1)
    result = compact.prepare(history)
    assert len(result[2].content) <= limit
    assert "[context truncated]" in result[2].content
    assert result[2].content[0] == text[0] and result[2].content[-1] == text[-1]
    call = result[2].tool_calls[0]
    assert call.id == "0" and call.name == "record"
    assert "_clamped" in call.arguments
    assert history[2].tool_calls[0].arguments == {"text": text}
    assert history[2].content == text
    assert compact.prepare(result) == result
    assert estimate_context_tokens(result) <= 2000
    assert_pairs(result)


def test_safe_cutoff_multiple_calls_reused_ids_and_followup_users():
    history = prefix() + pair(0, "x" * 200)
    history += [Message(role="user", content="follow up")]
    history += pair(0)
    history[-2].tool_calls.append(ToolCall(id="second", name="record", arguments={}))
    history.insert(len(history) - 1, Message(role="tool", tool_call_id="second", content="second result"))
    recent = history[4:]
    budget = estimate_context_tokens(prefix() + recent)
    result = manager(max_tokens=budget, keep_recent_turns=1).prepare(history)
    assert result == prefix() + recent
    assert_pairs(result)


def test_additional_system_and_pending_user_remain_protected():
    history = prefix() + pair(0, "x" * 1000)
    history += [Message(role="system", content="additional rules"), Message(role="user", content="new task")]
    expected = prefix() + history[-2:]
    result = manager(max_tokens=estimate_context_tokens(expected), keep_recent_turns=1).prepare(history)
    assert result == expected


@pytest.mark.parametrize("history", [
    prefix() + [Message(role="tool", tool_call_id="missing", content="malicious result")],
    prefix() + pair(0)[:1],
    prefix() + pair(0) + [pair(0)[1]],
    prefix() + [Message(role="user", tool_calls=[ToolCall(id="0", name="record")])],
    prefix() + [Message(role="assistant", tool_call_id="0")],
    prefix() + [Message(role="assistant", tool_calls=[ToolCall(id="0", name="record")] * 2)],
    prefix() + pair(0)[:1] + [Message(role="tool", tool_call_id="wrong")],
])
def test_rejects_orphan_duplicate_missing_and_forged_tool_metadata(history):
    with pytest.raises(ContextHistoryError):
        manager().prepare(history)


@pytest.mark.parametrize("role", ["system", "user"])
def test_protected_input_is_not_silently_truncated(role):
    history = prefix()
    history[0 if role == "system" else 1].content = "x" * 10_000
    with pytest.raises(ContextBudgetExceeded):
        manager(max_tokens=100, max_message_chars=64).prepare(history)
    assert len(history[0 if role == "system" else 1].content) == 10_000


@pytest.mark.parametrize("field", ["max_tokens", "max_message_chars", "keep_recent_turns"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        ContextConfig(**{field: value})


class FakeWorkspace:
    """The recording tool needs no workspace operations."""


class RecordingTool:
    name = "record"

    def __init__(self):
        self.arguments = []

    def definition(self):
        return ToolDefinition(name=self.name, description="Record arguments", parameters={"type": "object"})

    async def execute(self, arguments, context):
        self.arguments.append(arguments)
        return ToolResult(content="result " * 100)


async def test_agent_120_turns_bounds_every_request_and_retains_original_events():
    huge = "H" + " " * 100_000 + "T"
    responses = [ModelResponse(
        content=huge if index == 119 else f"turn {index}",
        tool_calls=[ToolCall(id=str(index), name="record", arguments={"value": huge if index == 119 else index})],
        usage=Usage(input_tokens=1, output_tokens=1),
    ) for index in range(120)] + [ModelResponse(content="done")]
    model = FakeModel(responses)
    requests = []
    original_complete = model.complete

    async def complete(messages, *, tools=None):
        assert estimate_context_tokens(messages, tools) <= 1500
        assert_pairs(messages)
        requests.append([message.model_copy(deep=True) for message in messages])
        return await original_complete(messages, tools=tools)

    model.complete = complete
    tool = RecordingTool()
    agent = Agent(model, FakeWorkspace(), policy=ActionPolicy(tool_risks={"record": Risk.LOW}),
                  tool_registry=ToolRegistry([tool]),
                  context_manager=manager(max_tokens=1500, max_message_chars=512, keep_recent_turns=2))
    state = await agent.run("task")
    assert state.status is RunStatus.COMPLETED
    assert state.model_calls == 121 and state.tool_calls == 120
    assert state.usage.input_tokens == 120 and state.usage.output_tokens == 120
    assert tool.arguments[-1] == {"value": huge}
    assert requests[-1][-4].tool_calls[0].id == "118"
    assert requests[-1][-2].tool_calls[0].id == "119"
    assert requests[-1][:2] == requests[0]
    events = agent.event_store.read(agent.run_id).events
    recorded_requests = [event.payload.messages for event in events if event.type == "ModelRequested"]
    assert recorded_requests == requests
    original = [event.payload.response for event in events if event.type == "ModelResponded"][-2]
    assert original.content == huge and original.tool_calls[0].arguments == {"value": huge}
    called = [event.payload.call for event in events if event.type == "ToolCalled"][-1]
    assert called.arguments == {"value": huge}


async def test_unfit_context_fails_before_model_call_or_budget_charge():
    model = FakeModel([ModelResponse(content="should not run")])
    model.complete = AsyncMock(wraps=model.complete)
    agent = Agent(model, FakeWorkspace(), context_manager=manager(max_tokens=10))
    with pytest.raises(ContextBudgetExceeded):
        await agent.run("x" * 1000)
    model.complete.assert_not_awaited()
    assert agent.state.status is RunStatus.FAILED
    assert agent.state.model_calls == agent.state.steps == 0
    events = agent.event_store.read(agent.run_id).events
    assert [event.type for event in events] == ["RunStarted", "RunFailed"]
    assert events[-1].payload.error_type == "ContextBudgetExceeded"