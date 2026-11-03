import os
from collections.abc import Callable, Iterator

import httpx
import pytest

from playground_llm.ollama import OllamaConfig, OllamaLLM

DEFAULT_LIVE_MODEL = "gemma4:12b"
LIVE_ENV = "LLM_LIVE_TESTS"
RAW_QUESTION = [{"role": "user", "content": "What is 17 times 23? Answer with the number only."}]


@pytest.fixture
def server() -> Iterator[OllamaLLM]:
    """실제 Ollama (모델은 테스트마다 확인). LIVE_ENV 를 켜지 않았거나 떠 있지 않으면 건너뛴다."""
    if os.environ.get(LIVE_ENV) != "1":
        pytest.skip(f"live 테스트는 GPU 를 쓴다 — {LIVE_ENV}=1 로 켠다")
    with OllamaLLM.connect(OllamaConfig.from_env()) as llm:
        try:
            llm._client.get("/api/tags")
        except httpx.HTTPError:
            pytest.skip("Ollama 가 떠 있지 않음")
        yield llm


@pytest.fixture
def require_model(server: OllamaLLM) -> Callable[[str], None]:
    """모델이 서버에 없으면 그 테스트를 건너뛰는 함수."""

    def require(model: str) -> None:
        names = {m["name"] for m in server._client.get("/api/tags").json()["models"]}
        if not {model, f"{model}:latest"} & names:  # 태그를 빼면 Ollama 는 :latest 로 받는다
            pytest.skip(f"{model} 이 Ollama 에 없음")

    return require


@pytest.fixture
def live_model() -> str:
    """live 테스트가 쓰는 모델 (OLLAMA_TEST_MODEL, 기본 DEFAULT_LIVE_MODEL)."""
    return os.environ.get("OLLAMA_TEST_MODEL", DEFAULT_LIVE_MODEL)


@pytest.fixture
def live(server: OllamaLLM, require_model: Callable[[str], None], live_model: str) -> OllamaLLM:
    """실제 Ollama. 떠 있지 않거나 live_model 이 없으면 건너뛴다."""
    require_model(live_model)
    return server


@pytest.fixture
def raw_thinking(
    server: OllamaLLM, require_model: Callable[[str], None]
) -> Callable[[str, bool | str], str]:
    """규칙표를 거치지 않고 think 값을 그대로 보내 thinking 을 받는 함수. 없으면 빈 문자열."""

    def ask(model: str, think: bool | str) -> str:
        require_model(model)
        body = {
            "model": model,
            "messages": RAW_QUESTION,
            "stream": False,
            "think": think,
            "options": {"temperature": 0, "seed": 1},
        }
        response = server._client.post("/api/chat", json=body)
        response.raise_for_status()
        return response.json()["message"].get("thinking", "")

    return ask
