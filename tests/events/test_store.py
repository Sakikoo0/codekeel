import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from codekeel.events.jsonl import JsonlEventStore
from codekeel.events.models import EVENT_ADAPTER, RunStarted, UnknownEvent, parse_event
from codekeel.events.store import EventStoreError, MemoryEventStore
from codekeel.models import Message
from codekeel.runtime import BudgetLimits


def event(sequence=1, *, event_id=None, run_id="test-run"):
    return RunStarted(
        event_id=event_id or f"event-{sequence}", run_id=run_id, sequence=sequence,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        payload={"messages": [Message(role="user", content="中文\n\x1b[31m")], "budgets": BudgetLimits()},
    )


@pytest.fixture(params=["memory", "jsonl"])
def store(request, tmp_path):
    return MemoryEventStore() if request.param == "memory" else JsonlEventStore(tmp_path)


def test_append_read_roundtrip_and_monotonic_sequence(store):
    expected = [event(i) for i in range(1, 4)]
    for item in expected:
        store.append(item)
    result = store.read("test-run")
    assert result.events == tuple(expected)
    assert result.warning is None
    assert [item.sequence for item in result.events] == [1, 2, 3]


@pytest.mark.parametrize("bad", [event(1), event(3), event(2, event_id="event-1")])
def test_rejects_duplicate_id_and_out_of_order_append_without_changing_trace(store, bad):
    store.append(event())
    with pytest.raises(EventStoreError):
        store.append(bad)
    assert store.read("test-run").events == (event(),)


def test_store_snapshot_cannot_be_changed_by_input_or_read_result_mutation(store):
    original = event()
    store.append(original)
    original.payload.messages[0].content = "changed after append"
    result = store.read("test-run")
    result.events[0].payload.messages.clear()
    assert store.read("test-run").events == (event(),)


def test_missing_read_does_not_create_trace(store, tmp_path):
    with pytest.raises(FileNotFoundError):
        store.read("missing")
    assert not (tmp_path / ".agent").exists()


@pytest.mark.parametrize("run_id", ["../outside", "/tmp/outside", "..", "a/b", "a\\b", "", "x\n", "x" * 129])
def test_invalid_run_id_is_rejected_before_file_access(store, run_id, tmp_path):
    with pytest.raises(ValueError):
        store.read(run_id)
    with pytest.raises(ValidationError):
        event(run_id=run_id)
    assert not (tmp_path / ".agent").exists()


def test_jsonl_appends_without_rewriting_existing_bytes(tmp_path):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    path = tmp_path / ".agent/runs/test-run/events.jsonl"
    prefix = path.read_bytes()
    JsonlEventStore(tmp_path).append(event(2))
    assert path.read_bytes().startswith(prefix)
    assert len(path.read_bytes().splitlines()) == 2
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("tail", [b'{"type":', b"invalid\n", b"\xff\n", b'{}\n'])
def test_corrupt_final_line_returns_prefix_with_warning_and_refuses_append(tmp_path, tail):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    path = tmp_path / ".agent/runs/test-run/events.jsonl"
    with path.open("ab") as stream:
        stream.write(tail)
    original = path.read_bytes()
    result = store.read("test-run")
    assert result.events == (event(),)
    assert result.warning == "Ignored corrupt final line 2"
    with pytest.raises(EventStoreError, match="Refusing"):
        store.append(event(2))
    assert path.read_bytes() == original


def test_valid_unterminated_final_line_remains_inspectable_but_not_appendable(tmp_path):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    path = tmp_path / ".agent/runs/test-run/events.jsonl"
    path.write_bytes(path.read_bytes().rstrip(b"\n"))
    result = store.read("test-run")
    assert result.events == (event(),)
    assert "no newline" in result.warning
    with pytest.raises(EventStoreError):
        store.append(event(2))


def test_interior_corruption_is_not_silently_skipped(tmp_path):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    path = tmp_path / ".agent/runs/test-run/events.jsonl"
    with path.open("ab") as stream:
        stream.write(b"not-json\n" + event(2).model_dump_json().encode() + b"\n")
    with pytest.raises(EventStoreError, match="line 2"):
        store.read("test-run")


@pytest.mark.parametrize("bad", [event(3), event(2, event_id="event-1"), event(2, run_id="other")])
def test_valid_json_with_invalid_order_or_run_is_hard_corruption_even_at_tail(tmp_path, bad):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    with (tmp_path / ".agent/runs/test-run/events.jsonl").open("a") as stream:
        stream.write(bad.model_dump_json() + "\n")
    with pytest.raises(EventStoreError):
        store.read("test-run")


def test_unknown_future_event_preserves_data_and_does_not_hide_following_events(tmp_path):
    store = JsonlEventStore(tmp_path)
    store.append(event())
    data = event(2).model_dump(mode="json") | {"type": "FutureEvent", "payload": [1, {"new": True}], "version": 7}
    with (tmp_path / ".agent/runs/test-run/events.jsonl").open("a") as stream:
        stream.write(json.dumps(data) + "\n")
    store.append(event(3))
    result = store.read("test-run")
    assert len(result.events) == 3
    assert isinstance(result.events[1], UnknownEvent)
    assert result.events[1].model_dump(mode="json") == data
    assert result.warning is None


def test_known_type_with_bad_payload_is_not_misclassified_as_unknown():
    with pytest.raises(ValidationError):
        parse_event(event().model_dump() | {"payload": {"invalid": "payload"}})
    assert EVENT_ADAPTER.json_schema()["discriminator"]["propertyName"] == "type"


@pytest.mark.parametrize(
    "component", [".agent", ".agent/runs", ".agent/runs/test-run", ".agent/runs/test-run/events.jsonl"],
)
def test_symlink_trace_components_cannot_escape_storage_root(tmp_path, component):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside
    if component.endswith("jsonl"):
        target = outside / "private"
        target.write_text("DO NOT CHANGE")
    link = root / component
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)
    store = JsonlEventStore(root)
    for operation in (lambda: store.read("test-run"), lambda: store.append(event())):
        with pytest.raises(OSError):
            operation()
    if target.is_file():
        assert target.read_text() == "DO NOT CHANGE"
    else:
        assert list(outside.iterdir()) == []


@pytest.mark.parametrize("kind", ["hardlink", "fifo"])
def test_trace_must_be_a_single_link_regular_file(tmp_path, kind):
    path = tmp_path / ".agent/runs/test-run/events.jsonl"
    path.parent.mkdir(parents=True)
    if kind == "hardlink":
        outside = tmp_path / "private"
        outside.write_text("SECRET")
        os.link(outside, path)
    else:
        os.mkfifo(path)
    with pytest.raises(EventStoreError, match="regular file"):
        JsonlEventStore(tmp_path).append(event())


def test_two_writers_cannot_append_the_same_sequence(tmp_path):
    JsonlEventStore(tmp_path).append(event())

    def append(index):
        try:
            JsonlEventStore(tmp_path).append(event(2, event_id=f"writer-{index}"))
            return True
        except EventStoreError:
            return False

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(append, [1, 2])) == [False, True]
    assert len(JsonlEventStore(tmp_path).read("test-run").events) == 2
