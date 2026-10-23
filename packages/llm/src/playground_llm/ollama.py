"""Ollama 네이티브 /api/chat 클라이언트 (`[ollama]` extras: httpx).

OpenAI 호환 경로(/v1)가 아니라 네이티브 API 를 쓴다 — `think` · `options.num_ctx` 를 넘기고
thinking 과 답을 따로 받을 수 있는 건 네이티브 쪽뿐이다. 패키지 루트는 이 모듈을 import 하지 않으므로
쓰는 쪽은 `from playground_llm.ollama import OllamaLLM` 으로 가져온다.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import httpx

from playground_llm.errors import GenerationError, GenerationTimeoutError, ModelNotFoundError
from playground_llm.types import (
    THINK,
    Generation,
    Message,
    ModelSpec,
    ParamValue,
    StreamDelta,
    Usage,
)

URL = "http://localhost:11434"
TIMEOUT_SEC = 120.0  # think on 은 thinking 토큰이 수천 개까지 간다. 첫 요청은 모델 적재 시간도 든다
# ModelSpec.params 중 options 가 아니라 요청 최상위로 가는 키
TOP_LEVEL_PARAMS = frozenset({"keep_alive"})
# 서버와 상관없는 이름 → Ollama options 이름 (프로필 표가 같은 이름을 쓰게)
OPTION_ALIASES = {"max_tokens": "num_predict"}
_NS_PER_MS = 1_000_000


@dataclass(frozen=True)
class OllamaConfig:
    """접속 설정."""

    url: str = URL
    timeout_sec: float = TIMEOUT_SEC
    """요청 하나의 읽기 시간 제한. 스트리밍은 조각 사이 간격에 걸린다."""

    @classmethod
    def from_env(cls) -> OllamaConfig:
        """OLLAMA_URL(없으면 OLLAMA_HOST) · OLLAMA_TIMEOUT 에서 읽는다."""
        default = cls()
        url = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST") or default.url
        if not url.startswith(("http://", "https://")):  # OLLAMA_HOST 는 host:port 로도 쓴다
            url = f"http://{url}"
        timeout = os.environ.get("OLLAMA_TIMEOUT")
        return cls(url=url, timeout_sec=float(timeout) if timeout else default.timeout_sec)


class OllamaLLM:
    """LLM Protocol 을 Ollama /api/chat 으로 채운다. `think` 는 항상 명시해서 보낸다.

    gemma4 처럼 thinking 을 지원하는 모델은 `think` 를 빼면 켜진 채로 돈다 (Ollama 0.40, gemma4:12b 확인).
    그래서 spec 에 think 가 없으면 false 를 보낸다.

    예)
        with OllamaLLM.connect(OllamaConfig.from_env()) as llm:
            spec = ModelSpec("ollama", "gemma4:12b", {"temperature": 0.1, "max_tokens": 512})
            llm.generate(spec, [Message("user", "조선의 수도는?")]).text
    """

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    @classmethod
    def connect(cls, config: OllamaConfig) -> OllamaLLM:
        """HTTP 클라이언트를 만든다. 실제 접속은 첫 요청 때. `with` 로 쓰면 끝날 때 닫는다."""
        timeout = httpx.Timeout(config.timeout_sec, connect=10.0)
        return cls(httpx.Client(base_url=config.url, timeout=timeout))

    def __enter__(self) -> OllamaLLM:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """연결을 닫는다."""
        self._client.close()

    def generate(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, object] | None = None,
    ) -> Generation:
        """대화에 이어질 답을 한 번에 받는다. json_schema 가 있으면 Ollama `format` 으로 넘긴다.

        Raises:
          ModelNotFoundError: 서버에 spec.name 모델이 없을 때.
          GenerationTimeoutError: 시간 제한을 넘겼을 때.
          GenerationError: 연결 실패 · 오류 응답 · 응답 형식이 다를 때.
        """
        start = time.perf_counter()
        with _errors(spec):
            request = chat_request(spec, messages, stream=False, json_schema=json_schema)
            response = self._client.post("/api/chat", json=request)
            _raise_for_status(response, spec)
            body = response.json()
        message = body.get("message") or {}
        return Generation(
            text=message.get("content", ""),
            model=spec,
            usage=_usage(body, start),
            thinking=message.get("thinking") or None,
        )

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        """조각(StreamDelta)을 오는 대로 보내고, 끝나면 전체 Generation 을 보낸다.

        Raises:
          generate 와 같다. 조각을 몇 개 보낸 뒤에도 날 수 있다.
        """
        start = time.perf_counter()
        text: list[str] = []
        thinking: list[str] = []
        final: Mapping[str, object] = {}
        with _errors(spec):
            request = chat_request(spec, messages, stream=True)
            with self._client.stream("POST", "/api/chat", json=request) as response:
                if response.is_error:
                    response.read()
                    _raise_for_status(response, spec)
                for line in response.iter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if "error" in chunk:  # 스트림 중간에 서버가 실패를 알린다
                        raise GenerationError(f"{spec.name}: {chunk['error']}")
                    message = chunk.get("message") or {}
                    if piece := message.get("thinking"):
                        thinking.append(piece)
                        yield StreamDelta("thinking", piece)
                    if piece := message.get("content"):
                        text.append(piece)
                        yield StreamDelta("text", piece)
                    if chunk.get("done"):
                        final = chunk
        if not final:
            raise GenerationError(f"{spec.name}: stream ended without a final chunk")
        yield Generation(
            text="".join(text),
            model=spec,
            usage=_usage(final, start),
            thinking="".join(thinking) or None,
        )


def chat_request(
    spec: ModelSpec,
    messages: Sequence[Message],
    *,
    stream: bool,
    json_schema: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """/api/chat 요청 본문. params 를 options · 최상위로 나누고 think 를 늘 넣는다."""
    if not messages:
        raise ValueError("messages must not be empty")
    options: dict[str, object] = {}
    top: dict[str, object] = {}
    for key, value in spec.params.items():
        if key == THINK:
            continue
        target = top if key in TOP_LEVEL_PARAMS else options
        target[OPTION_ALIASES.get(key, key)] = _json_value(value)
    return {
        "model": spec.name,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "stream": stream,
        THINK: spec.think,
        **({"options": options} if options else {}),
        **({"format": dict(json_schema)} if json_schema is not None else {}),
        **top,
    }


def _json_value(value: ParamValue) -> object:
    return list(value) if isinstance(value, tuple) else value  # Ollama stop 은 JSON 배열


def _usage(body: Mapping[str, object], start: float) -> Usage:
    def ms(key: str) -> int | None:
        value = body.get(key)
        return round(value / _NS_PER_MS) if isinstance(value, int) else None

    def count(key: str) -> int | None:
        value = body.get(key)
        return value if isinstance(value, int) else None

    return Usage(
        prompt_tokens=count("prompt_eval_count") or 0,
        completion_tokens=count("eval_count") or 0,
        total_ms=round((time.perf_counter() - start) * 1000),
        cached_prompt_tokens=count("prompt_eval_cached_count"),
        load_ms=ms("load_duration"),
        prompt_eval_ms=ms("prompt_eval_duration"),
        eval_ms=ms("eval_duration"),
    )


def _raise_for_status(response: httpx.Response, spec: ModelSpec) -> None:
    if not response.is_error:
        return
    try:
        detail = str(response.json().get("error", response.text))
    except ValueError:
        detail = response.text
    if response.status_code == httpx.codes.NOT_FOUND and "not found" in detail:
        raise ModelNotFoundError(f"{spec.name}: {detail}")
    raise GenerationError(f"{spec.name}: HTTP {response.status_code} {detail}")


@contextmanager
def _errors(spec: ModelSpec) -> Iterator[None]:
    """HTTP 클라이언트(httpx) 예외를 playground_llm 예외로 바꾼다."""
    try:
        yield
    except httpx.TimeoutException as e:
        raise GenerationTimeoutError(f"{spec.name}: {type(e).__name__}") from e
    except httpx.HTTPError as e:
        raise GenerationError(f"{spec.name}: {type(e).__name__}: {e}") from e
    except json.JSONDecodeError as e:
        raise GenerationError(f"{spec.name}: invalid JSON from server") from e
