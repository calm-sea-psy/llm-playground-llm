"""OllamaLLM 테스트. 단위 테스트는 httpx.MockTransport, 마지막 하나는 실제 Ollama (없으면 건너뜀)."""

import json
import os
import subprocess
import sys

import httpx
import pytest

from playground_llm import (
    Generation,
    GenerationError,
    GenerationTimeoutError,
    Message,
    ModelNotFoundError,
    ModelSpec,
    StreamDelta,
)
from playground_llm.ollama import OllamaConfig, OllamaLLM, chat_request

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
        SPEC.with_params(think=True, stop=("\n\n",), keep_alive="10m"), MESSAGES, stream=True
    )

    assert off["think"] is False  # 빼면 gemma4 는 thinking 이 켜진 채로 돈다
    assert off["options"] == {"temperature": 0.1, "num_predict": 64}
    assert off["messages"][1] == {"role": "user", "content": "17 곱하기 23 은?"}
    assert on["think"] is True
    assert on["stream"] is True
    assert on["options"]["stop"] == ["\n\n"]
    assert on["keep_alive"] == "10m"
    assert "think" not in on["options"]


def test_generate_splits_thinking_and_text_and_reads_usage():
    body = {"message": {"role": "assistant", "content": "391", "thinking": "17×23"}, **FINAL}
    llm, requests = _llm(lambda _: httpx.Response(200, json=body))

    generation = llm.generate(SPEC.with_params(think=True), MESSAGES)

    assert (generation.text, generation.thinking) == ("391", "17×23")
    assert requests[0]["think"] is True
    usage = generation.usage
    assert (usage.prompt_tokens, usage.completion_tokens, usage.cached_prompt_tokens) == (27, 4, 25)
    assert (usage.load_ms, usage.prompt_eval_ms, usage.eval_ms) == (6, 82, 111)
    assert usage.thinking_tokens is None
    assert usage.total_ms >= 0


def test_generate_without_thinking_field_gives_none():
    body = {"message": {"role": "assistant", "content": "391"}, **FINAL}
    llm, _ = _llm(lambda _: httpx.Response(200, json=body))

    assert llm.generate(SPEC, MESSAGES).thinking is None


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

    events = list(llm.stream(SPEC.with_params(think=True), MESSAGES))

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


def test_server_error_and_timeout_map_to_generation_errors():
    failing, _ = _llm(lambda _: httpx.Response(500, json={"error": "out of memory"}))

    def time_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    slow, _ = _llm(time_out)

    with pytest.raises(GenerationError, match="500 out of memory"):
        failing.generate(SPEC, MESSAGES)
    with pytest.raises(GenerationTimeoutError):
        slow.generate(SPEC, MESSAGES)
    assert issubclass(GenerationTimeoutError, TimeoutError)


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


@pytest.fixture
def live() -> OllamaLLM:
    """실제 Ollama. 떠 있지 않거나 모델이 없으면 건너뛴다 (OLLAMA_TEST_MODEL, 기본 gemma4:12b)."""
    llm = OllamaLLM.connect(OllamaConfig.from_env())
    model = os.environ.get("OLLAMA_TEST_MODEL", "gemma4:12b")
    try:
        names = {m["name"] for m in llm._client.get("/api/tags").json()["models"]}
    except httpx.HTTPError:
        pytest.skip("Ollama 가 떠 있지 않음")
    if model not in names:
        pytest.skip(f"{model} 이 Ollama 에 없음")
    return llm


def test_live_think_off_returns_no_thinking(live: OllamaLLM):
    spec = ModelSpec(
        "ollama", os.environ.get("OLLAMA_TEST_MODEL", "gemma4:12b"), {"temperature": 0}
    )
    question = [Message("user", "17 곱하기 23 은? 숫자만 답해.")]

    off = live.generate(spec, question)
    streamed = list(live.stream(spec.with_params(think=True), question))

    assert off.thinking is None
    assert "391" in off.text
    final = streamed[-1]
    assert isinstance(final, Generation)
    assert final.thinking  # 켜면 thinking 이 따로 온다
    assert "391" in final.text
    assert final.usage.completion_tokens > off.usage.completion_tokens


def test_request_sends_json_schema_as_format():
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}

    request = chat_request(SPEC, MESSAGES, stream=False, json_schema=schema)

    assert request["format"] == schema
    assert "format" not in chat_request(SPEC, MESSAGES, stream=False)
