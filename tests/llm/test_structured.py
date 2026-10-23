import json
import os

import httpx
import pytest
from pydantic import BaseModel, Field

from playground_llm import FakeLLM, Message, ModelSpec, StructuredOutputError, Think
from playground_llm.ollama import OllamaConfig, OllamaLLM
from playground_llm.structured import generate_json

SPEC = ModelSpec("fake", "m")
QUESTION = [Message("user", "조선의 수도와 확신도를 JSON 으로")]


class Answer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1)


def test_returns_validated_value_and_sends_schema():
    llm = FakeLLM(['{"answer": "한양", "confidence": 0.9}'])

    result = generate_json(llm, SPEC, QUESTION, Answer)

    assert result.value == Answer(answer="한양", confidence=0.9)
    assert result.attempts == 1
    assert llm.calls[0].json_schema == Answer.model_json_schema()


def test_retries_with_validation_errors_then_succeeds():
    llm = FakeLLM(
        ['{"answer": "한양", "confidence": 1.5}', '{"answer": "한양", "confidence": 0.8}']
    )

    result = generate_json(llm, SPEC, QUESTION, Answer)

    assert result.value.confidence == 0.8
    assert result.attempts == 2
    retry = llm.calls[1].messages
    assert retry[:1] == tuple(QUESTION)
    assert retry[1] == Message("assistant", '{"answer": "한양", "confidence": 1.5}')
    assert retry[2].role == "user"
    assert "confidence" in retry[2].content  # 어느 필드가 틀렸는지 보여 준다


def test_gives_up_after_retries_with_last_text():
    llm = FakeLLM(["한양입니다"])

    with pytest.raises(StructuredOutputError, match="after 2 attempts") as caught:
        generate_json(llm, SPEC, QUESTION, Answer, retries=1)

    assert caught.value.last_text == "한양입니다"
    assert caught.value.attempts == 2
    assert len(llm.calls) == 2


def test_no_retry_when_retries_is_zero():
    llm = FakeLLM(["{}"])

    with pytest.raises(StructuredOutputError):
        generate_json(llm, SPEC, QUESTION, Answer, retries=0)

    assert len(llm.calls) == 1


def test_thinking_is_kept_apart_from_parsed_text():
    llm = FakeLLM(['{"answer": "한양", "confidence": 0.9}'])

    result = generate_json(llm, SPEC.with_params(think=Think.ON), QUESTION, Answer)

    assert result.generation.thinking == FakeLLM.THINKING
    assert result.value.answer == "한양"


def test_live_ollama_follows_schema():
    model = os.environ.get("OLLAMA_TEST_MODEL", "gemma4:12b")
    llm = OllamaLLM.connect(OllamaConfig.from_env())
    try:
        names = {m["name"] for m in json.loads(llm._client.get("/api/tags").text)["models"]}
    except httpx.HTTPError:
        pytest.skip("Ollama 가 떠 있지 않음")
    if model not in names:
        pytest.skip(f"{model} 이 Ollama 에 없음")
    spec = ModelSpec("ollama", model, {"temperature": 0})
    question = [Message("user", "조선의 첫 수도는? answer 에 지명만, confidence 는 0~1.")]

    result = generate_json(llm, spec, question, Answer, retries=0)

    assert result.attempts == 1
    assert result.value.answer
    assert 0 <= result.value.confidence <= 1
