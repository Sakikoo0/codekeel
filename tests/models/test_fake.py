import pytest

from codekeel.models import FakeModel, Message, ModelResponse, Usage


async def test_fake_model_returns_scripted_responses() -> None:
    first = ModelResponse(content="first", usage=Usage(input_tokens=2, output_tokens=1))
    second = ModelResponse(content="second", usage=Usage(input_tokens=3, output_tokens=2))
    model = FakeModel([first, second])
    messages = [Message(role="user", content="hello")]

    assert await model.complete(messages) == first
    assert await model.complete(messages) == second


async def test_fake_model_raises_when_script_exhausted() -> None:
    model = FakeModel([])

    with pytest.raises(RuntimeError, match="no responses remaining"):
        await model.complete([])
