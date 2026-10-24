from collections.abc import Sequence

import pytest

from playground_llm import LLM, FakeLLM, Generation, Message, ModelSpec, StreamDelta, Think

SPEC = ModelSpec("fake", "m")
QUESTION = [Message("system", "위키 근거로만 답한다"), Message("user", "조선의 수도는?")]


def test_fake_returns_replies_in_order_then_repeats_last():
    llm = FakeLLM(["첫 답", "둘째 답"])

    texts = [llm.generate(SPEC, QUESTION).text for _ in range(3)]

    assert texts == ["첫 답", "둘째 답", "둘째 답"]


def test_fake_records_calls_with_spec_and_messages():
    llm = FakeLLM(["답"])
    on = SPEC.with_params(think=Think.ON)

    llm.generate(SPEC, QUESTION)
    llm.generate(on, QUESTION[1:])

    assert [call.spec for call in llm.calls] == [SPEC, on]
    assert llm.calls[1].messages == (QUESTION[1],)


def test_fake_fills_thinking_only_when_think_is_on():
    llm = FakeLLM(["서울 입니다"])

    off = llm.generate(SPEC.with_params(think=Think.OFF), QUESTION)  # OFF 도 StrEnum 이라 참이다
    on = llm.generate(SPEC.with_params(think=Think.ON), QUESTION)

    assert off.thinking is None
    assert on.thinking == FakeLLM.THINKING
    assert on.text == "서울 입니다"  # thinking 은 text 에 섞이지 않는다
    assert on.usage.completion_tokens > off.usage.completion_tokens == 2


def test_fake_generation_always_stops():
    long_answer = "가 " * 50
    llm = FakeLLM([long_answer])

    generation = llm.generate(SPEC.with_params(max_tokens=4), QUESTION)

    assert generation.done_reason == "stop"
    assert generation.text == long_answer  # max_tokens 로 자르지 않는다


def test_fake_reply_can_depend_on_input():
    def echo(spec: ModelSpec, messages: Sequence[Message]) -> str:
        return f"{spec.name}:{messages[-1].content}"

    assert FakeLLM([echo]).generate(SPEC, QUESTION).text == "m:조선의 수도는?"


def test_fake_satisfies_llm_protocol():
    llm: LLM = FakeLLM(["답"])

    assert llm.generate(SPEC, QUESTION).model == SPEC


def test_fake_rejects_empty_inputs():
    with pytest.raises(ValueError, match="replies"):
        FakeLLM([])
    with pytest.raises(ValueError, match="messages"):
        FakeLLM(["답"]).generate(SPEC, [])


def test_fake_stream_sends_thinking_then_text_pieces_then_generation():
    llm = FakeLLM(["서울 입니다"])

    events = list(llm.stream(SPEC.with_params(think=Think.ON), QUESTION))

    deltas = [e for e in events[:-1] if isinstance(e, StreamDelta)]
    final = events[-1]
    assert isinstance(final, Generation)
    assert "".join(d.text for d in deltas if d.kind == "thinking") == FakeLLM.THINKING
    assert "".join(d.text for d in deltas if d.kind == "text") == "서울 입니다"
    assert [d.kind for d in deltas] == sorted(
        (d.kind for d in deltas), key=lambda kind: kind != "thinking"
    )
    assert len(llm.calls) == 1
