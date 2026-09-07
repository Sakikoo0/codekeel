"""Deterministic clamp, old-result eviction, and pair-safe sliding window."""

import json

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from coding_agent.context.manager import ContextBudgetExceeded, ContextHistoryError
from coding_agent.models.base import Message, ToolDefinition, ToolResult

_CLAMP_MARKER = "\n[context truncated]\n"
_CLEARED = "[tool result cleared]"


class ContextConfig(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    max_tokens: int = Field(default=32_000, gt=0)
    max_message_chars: int = Field(default=16_000, ge=64)
    keep_recent_turns: int = Field(default=4, ge=1)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def estimate_context_tokens(messages: list[Message], tools: list[ToolDefinition] | None = None) -> int:
    """Four serialized characters per token; a deterministic heuristic, not a tokenizer.

    Includes roles, call IDs, arguments and tool schemas. Callers must leave room
    for provider-specific formatting, tokenizer differences and generated output.
    """
    serialized = _json({
        "messages": [message.model_dump(mode="json") for message in messages],
        "tools": [tool.model_dump(mode="json") for tool in tools or []],
    })
    return (len(serialized) + 3) // 4


def _clamp(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    remaining = limit - len(_CLAMP_MARKER)
    head = (remaining + 1) // 2
    tail = remaining // 2
    return text[:head] + _CLAMP_MARKER + text[-tail:]


def _turns(messages: list[Message]) -> list[list[int]]:
    """Pin every system message and the first user task; group remaining turns.

    A turn is an assistant message plus all its results and any preceding follow-up
    user messages. Trailing user messages form a pending turn. Reused call IDs in
    separate completed exchanges are valid; duplicates within an exchange are not.
    """
    turns: list[list[int]] = []
    pending: list[int] = []
    task_seen = False
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.tool_call_id is not None or message.role == "tool":
            raise ContextHistoryError("Orphan tool result or misplaced tool_call_id")
        if message.tool_calls and message.role != "assistant":
            raise ContextHistoryError("Only assistant messages may contain tool calls")
        if message.role == "system":
            index += 1
            continue
        if message.role == "user" and not task_seen:
            task_seen = True
            index += 1
            continue
        pending.append(index)
        index += 1
        if message.role == "user":
            continue
        ids = {call.id for call in message.tool_calls}
        if len(ids) != len(message.tool_calls):
            raise ContextHistoryError("Duplicate tool call ID within an exchange")
        while ids:
            if index >= len(messages):
                raise ContextHistoryError("Missing tool result")
            result = messages[index]
            if result.role != "tool" or result.tool_calls or result.tool_call_id not in ids:
                raise ContextHistoryError("Missing, duplicate or mismatched tool result")
            ids.remove(result.tool_call_id)
            pending.append(index)
            index += 1
        turns.append(pending)
        pending = []
    if pending:
        turns.append(pending)
    return turns


def clamp_history(messages: list[Message], config: ContextConfig) -> list[Message]:
    """Copy history and bound pathological assistant parts without evicting turns."""
    history = [message.model_copy(deep=True) for message in messages]
    for message in history:
        if message.role == "assistant":
            if message.content is not None:
                message.content = _clamp(message.content, config.max_message_chars)
            for call in message.tool_calls:
                arguments = _json(call.arguments)
                if len(arguments) > config.max_message_chars:
                    # Historical arguments only: execution has already happened.
                    limit = config.max_message_chars
                    replacement = {"_clamped": _clamp(arguments, limit)}
                    while len(_json(replacement)) > config.max_message_chars:
                        limit = max(len(_CLAMP_MARKER) + 2, limit // 2)
                        replacement = {"_clamped": _clamp(arguments, limit)}
                    call.arguments = replacement
    return history


class DeterministicContextManager:
    """Reduce history without I/O or model calls; never alter the caller's objects."""

    def __init__(self, config: ContextConfig | None = None) -> None:
        self.config = config if config is not None else ContextConfig()

    def prepare(
        self, messages: list[Message], *, tools: list[ToolDefinition] | None = None,
    ) -> list[Message]:
        turns = _turns(messages)
        history = clamp_history(messages, self.config)

        def fits(candidate: list[Message]) -> bool:
            return estimate_context_tokens(candidate, tools) <= self.config.max_tokens

        if fits(history):
            return history
        old_turns = turns[:-self.config.keep_recent_turns]
        for turn in old_turns:
            for index in turn:
                message = history[index]
                if message.role != "tool" or message.content is None:
                    continue
                try:
                    result = ToolResult.model_validate_json(message.content)
                except ValidationError:
                    replacement = _CLEARED
                else:
                    replacement = result.model_copy(update={"content": _CLEARED}).model_dump_json()
                if len(_json(replacement)) < len(_json(message.content)):
                    message.content = replacement
            if fits(history):
                return history
        removed: set[int] = set()
        for turn in old_turns:
            removed.update(turn)
            candidate = [message for index, message in enumerate(history) if index not in removed]
            if fits(candidate):
                return candidate
        raise ContextBudgetExceeded("System, task, recent turns and tool definitions exceed the context budget")