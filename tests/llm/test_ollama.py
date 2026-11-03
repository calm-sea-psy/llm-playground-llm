"""OllamaLLM 테스트. 단위 테스트는 httpx.MockTransport, 마지막 하나는 실제 Ollama (없으면 건너뜀)."""

import json
import subprocess
import sys
from collections.abc import Callable

import httpx
import pytest

from playground_llm import (
    Generation,
    GenerationError,
    GenerationTimeoutError,
    LLMError,
    Message,
    ModelNotFoundError,
    ModelSpec,
    StreamDelta,
    Think,
)
from playground_llm.ollama import (
    THINK_TO_OLLAMA,
    OllamaConfig,
    OllamaLLM,
    ThinkRule,
    ThinkRules,
    chat_request,
)

SPEC = ModelSpec("ollama", "gemma4:12b", {"temperature": 0.1, "max_tokens": 64})
MESSAGES = [Message("system", "짧게 답한다"), Message("user", "17 곱하기 23 은?")]
FINAL = {
    "done": True,
    "prompt_eval_count": 27,
    "prompt_eval_cached_count": 25,
    "eval_count": 4,
    "load_duration": 6_456_200,
    "prompt_eval_duration": 82_327_000,
    "eval_duration": 110_883_000,
}


def _llm(handler) -> tuple[OllamaLLM, list[dict]]:
    requests: list[dict] = []

    def record(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return handler(request)

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(record))
    return OllamaLLM(client), requests


def test_request_always_sends_think_and_maps_params():
    off = chat_request(SPEC, MESSAGES, stream=False)
    on = chat_request(
        SPEC.with_params(think=Think.ON, stop=("\n\n",), keep_alive="10m"), MESSAGES, stream=True
    )

    assert off["think"] is False  # 빼면 gemma4 는 thinking 이 켜진 채로 돈다
    assert off["options"] == {"temperature": 0.1, "num_predict": 64}
    assert off["messages"][1] == {"role": "user", "content": "17 곱하기 23 은?"}
    assert on["think"] is True
    assert on["stream"] is True
    assert on["options"]["stop"] == ["\n\n"]
    assert on["keep_alive"] == "10m"
    assert "think" not in on["options"]


def test_request_sends_think_level_as_string():
    spec = ModelSpec("ollama", "gpt-oss:20b", {"think": Think.LOW})

    assert chat_request(spec, MESSAGES, stream=False)["think"] == "low"


def test_think_map_covers_every_think_value():
    assert set(THINK_TO_OLLAMA) == set(Think)


@pytest.mark.parametrize(
    ("model", "think"),
    [
        ("gpt-oss:20b", Think.OFF),  # false 로 꺼지지 않는다
        ("gpt-oss:120b", Think.ON),  # true 는 medium 과 같다
        ("gemma4:e4b", Think.ON),  # true 여도 thinking 이 오지 않는다
        ("gemma4:12b", Think.LOW),  # 정도를 무시하고 true 로 돈다 (기본 규칙)
    ],
)
def test_request_rejects_think_the_model_does_not_take(model: str, think: Think):
    spec = ModelSpec("ollama", model, {"think": think})

    with pytest.raises(ValueError, match=f"think={think} is not supported"):
        chat_request(spec, MESSAGES, stream=False)


def test_rejection_names_allowed_values_and_sends_nothing():
    llm, requests = _llm(lambda _: httpx.Response(200, json=FINAL))

    with pytest.raises(ValueError, match="use one of: low, medium, high"):
        llm.generate(ModelSpec("ollama", "gpt-oss:20b"), MESSAGES)  # think 를 빼면 OFF

    assert requests == []


def test_think_rules_reject_shadowed_rule_and_missing_default():
    general = ThinkRule(prefix="gemma4", allowed=frozenset({Think.OFF}), reason="r")
    specific = ThinkRule(prefix="gemma4:e4b", allowed=frozenset({Think.OFF}), reason="r")
    default = ThinkRule(prefix="", allowed=frozenset({Think.OFF, Think.ON}), reason="r")

    ThinkRules((specific, general, default))  # 구체적인 규칙이 앞이면 된다
    with pytest.raises(ValueError, match="shadowed"):
        ThinkRules((general, specific, default))
    with pytest.raises(ValueError, match="default rule"):
        ThinkRules((specific, general))


def test_generate_splits_thinking_and_text_and_reads_usage():
    body = {"message": {"role": "assistant", "content": "391", "thinking": "17×23"}, **FINAL}
    llm, requests = _llm(lambda _: httpx.Response(200, json=body))

    generation = llm.generate(SPEC.with_params(think=Think.ON), MESSAGES)

    assert (generation.text, generation.thinking) == ("391", "17×23")
    assert requests[0]["think"] is True
    usage = generation.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cached_prompt_tokens) == (27, 4, 25)
    assert (usage.load_ms, usage.prompt_eval_ms, usage.eval_ms) == (6, 82, 111)
    assert usage.total_ms >= 0


def test_generate_without_thinking_field_gives_none():
    body = {"message": {"role": "assistant", "content": "391"}, **FINAL}
    llm, _ = _llm(lambda _: httpx.Response(200, json=body))

    assert llm.generate(SPEC, MESSAGES).thinking is None


@pytest.mark.parametrize(
    ("server", "expected"),
    [("stop", "stop"), ("length", "length"), (None, None), ("unload", None)],
)
def test_generate_and_stream_read_done_reason(server: str | None, expected: str | None):
    final = {**FINAL, "done_reason": server} if server else FINAL
    body = {"message": {"role": "assistant", "content": "391"}, **final}
    lines = [{"message": {"content": "391"}}, {"message": {"content": ""}, **final}]
    payload = "\n".join(json.dumps(line) for line in lines) + "\n"
    generate_llm, _ = _llm(lambda _: httpx.Response(200, json=body))
    stream_llm, _ = _llm(lambda _: httpx.Response(200, content=payload.encode()))

    streamed = list(stream_llm.stream(SPEC, MESSAGES))[-1]

    assert generate_llm.generate(SPEC, MESSAGES).done_reason == expected
    assert isinstance(streamed, Generation)
    assert streamed.done_reason == expected


def test_stream_yields_thinking_then_text_then_generation():
    lines = [
        {"message": {"thinking": "17×20=340, "}},
        {"message": {"thinking": "17×3=51"}},
        {"message": {"content": "39"}},
        {"message": {"content": "1"}},
        {"message": {"content": ""}, **FINAL},
    ]
    payload = "\n".join(json.dumps(line, ensure_ascii=False) for line in lines) + "\n"
    llm, requests = _llm(lambda _: httpx.Response(200, content=payload.encode()))

    events = list(llm.stream(SPEC.with_params(think=Think.ON), MESSAGES))

    deltas = [e for e in events if isinstance(e, StreamDelta)]
    assert [(d.kind, d.text) for d in deltas] == [
        ("thinking", "17×20=340, "),
        ("thinking", "17×3=51"),
        ("text", "39"),
        ("text", "1"),
    ]
    final = events[-1]
    assert isinstance(final, Generation)
    assert (final.text, final.thinking) == ("391", "17×20=340, 17×3=51")
    assert final.usage.completion_tokens == 4
    assert requests[0]["stream"] is True


def test_stream_error_chunk_raises():
    payload = json.dumps({"message": {"content": "3"}}) + "\n" + json.dumps({"error": "boom"})
    llm, _ = _llm(lambda _: httpx.Response(200, content=payload.encode()))

    with pytest.raises(GenerationError, match="boom"):
        list(llm.stream(SPEC, MESSAGES))


def test_missing_model_raises_model_not_found():
    llm, _ = _llm(lambda _: httpx.Response(404, json={"error": "model 'nope' not found"}))

    with pytest.raises(ModelNotFoundError, match="not found"):
        llm.generate(SPEC, MESSAGES)
    with pytest.raises(ModelNotFoundError):
        list(llm.stream(SPEC, MESSAGES))


@pytest.mark.parametrize(
    "body",
    [
        ["not", "an", "object"],
        {"message": "not an object"},
        {"message": {"content": 391}},
        {"message": {"content": "391", "thinking": 1}},
    ],
    ids=["body-list", "message-str", "content-int", "thinking-int"],
)
def test_malformed_response_raises_generation_error(body: object):
    payload = json.dumps(body).encode()
    generate_llm, _ = _llm(lambda _: httpx.Response(200, json=body))
    stream_llm, _ = _llm(lambda _: httpx.Response(200, content=payload))

    with pytest.raises(GenerationError):
        generate_llm.generate(SPEC, MESSAGES)
    with pytest.raises(GenerationError):
        list(stream_llm.stream(SPEC, MESSAGES))


def test_null_text_fields_count_as_absent():
    body = {"message": {"content": None, "thinking": None}}
    generate_llm, _ = _llm(lambda _: httpx.Response(200, json=body))
    stream_body = b'{"message": {"content": null}}\n{"message": {"content": "391"}, "done": true}\n'
    stream_llm, _ = _llm(lambda _: httpx.Response(200, content=stream_body))

    generation = generate_llm.generate(SPEC, MESSAGES)
    final = list(stream_llm.stream(SPEC, MESSAGES))[-1]

    assert generation.text == ""
    assert generation.thinking is None
    assert isinstance(final, Generation)
    assert final.text == "391"


def test_generate_without_message_gives_empty_text():
    llm, _ = _llm(lambda _: httpx.Response(200, json=FINAL))

    assert llm.generate(SPEC, MESSAGES).text == ""


def test_error_response_with_non_object_json_keeps_body_as_detail():
    llm, _ = _llm(lambda _: httpx.Response(500, json=["boom"]))

    with pytest.raises(GenerationError, match="boom"):
        llm.generate(SPEC, MESSAGES)


def test_server_error_and_timeout_map_to_generation_errors():
    failing, _ = _llm(lambda _: httpx.Response(500, json={"error": "out of memory"}))

    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    slow, _ = _llm(time_out)

    with pytest.raises(GenerationError, match="500 out of memory"):
        failing.generate(SPEC, MESSAGES)
    with pytest.raises(GenerationTimeoutError):
        slow.generate(SPEC, MESSAGES)


def test_errors_also_inherit_builtin_exceptions():
    assert issubclass(GenerationTimeoutError, TimeoutError)
    assert issubclass(GenerationTimeoutError, RuntimeError)
    assert issubclass(GenerationError, RuntimeError)
    assert issubclass(ModelNotFoundError, LookupError)
    assert issubclass(GenerationError, LLMError)
    assert issubclass(ModelNotFoundError, LLMError)


def test_config_from_env_accepts_host_without_scheme(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:11434")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "30")

    assert OllamaConfig.from_env() == OllamaConfig(url="http://127.0.0.1:11434", timeout_sec=30.0)


def test_package_root_does_not_import_extras():
    # 타입 · 프롬프트만 쓰는 쪽은 [ollama] · [json] extras 없이 import 할 수 있어야 한다
    code = "import sys, playground_llm; print('httpx' in sys.modules or 'pydantic' in sys.modules)"
    result = subprocess.run(  # noqa: S603  # 고정 인자로 현재 인터프리터 실행
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"


def test_live_think_off_returns_no_thinking(live: OllamaLLM, live_model: str):
    spec = ModelSpec("ollama", live_model, {"temperature": 0})
    question = [Message("user", "17 곱하기 23 은? 숫자만 답해.")]

    off = live.generate(spec, question)
    streamed = list(live.stream(spec.with_params(think=Think.ON), question))

    assert off.thinking is None
    assert "391" in off.text
    assert off.done_reason == "stop"
    final = streamed[-1]
    assert isinstance(final, Generation)
    assert final.thinking  # 켜면 thinking 이 따로 온다
    assert "391" in final.text
    assert final.usage.completion_tokens > off.usage.completion_tokens


def test_live_done_reason_is_length_when_max_tokens_cuts_the_answer(
    live: OllamaLLM, live_model: str
):
    spec = ModelSpec("ollama", live_model, {"temperature": 0, "max_tokens": 8})
    question = [Message("user", "조선의 왕 27명을 순서대로 모두 설명해 줘.")]

    cut = live.generate(spec, question)
    streamed = list(live.stream(spec, question))[-1]

    assert cut.done_reason == "length"
    assert cut.usage.completion_tokens == 8
    assert isinstance(streamed, Generation)
    assert streamed.done_reason == "length"


# 아래 test_live_rule_* 는 THINK_RULES 의 근거가 지금 서버에서도 맞는지 본다.
# 깨지면 Ollama 동작이 바뀐 것 — 규칙표와 THINK_RULES_OLLAMA_VERSION 을 다시 잰다
def test_live_rule_gpt_oss_false_still_thinks_and_true_equals_medium(
    raw_thinking: Callable[[str, bool | str], str],
):
    model = "gpt-oss:20b"

    assert raw_thinking(model, False)  # → OFF 거부
    assert raw_thinking(model, True) == raw_thinking(model, "medium")  # → ON 거부


def test_live_rule_gemma4_e4b_never_returns_thinking(
    raw_thinking: Callable[[str, bool | str], str],
):
    assert raw_thinking("gemma4:e4b", True) == ""  # → ON 거부


def test_live_rule_levels_are_ignored_on_unlisted_models(
    raw_thinking: Callable[[str, bool | str], str],
):
    on = raw_thinking("gemma4:12b", True)

    assert on
    assert raw_thinking("gemma4:12b", "low") == on  # → 기본 규칙은 OFF · ON 만


def test_live_server_rejects_think_on_for_model_without_thinking(
    server: OllamaLLM, require_model: Callable[[str], None]
):
    # 표에 없는 모델도 thinking 이 없으면 서버가 400 으로 막는다 → 기본 규칙이 ON 을 서버에 맡겨도 된다
    require_model("phi4-mini")
    spec = ModelSpec("ollama", "phi4-mini", {"think": Think.ON})

    with pytest.raises(GenerationError, match="does not support thinking"):
        server.generate(spec, MESSAGES)


def test_request_sends_json_schema_as_format():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    request = chat_request(SPEC, MESSAGES, stream=False, json_schema=schema)

    assert request["format"] == schema
    assert "format" not in chat_request(SPEC, MESSAGES, stream=False)
