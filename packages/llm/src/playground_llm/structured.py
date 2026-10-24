"""구조화 출력: pydantic 모델로 JSON Schema 를 만들어 서버에 요구하고, 답을 검증한다 (`[json]` extras).

LLM 출력을 파싱하는 경계 모듈이라 pydantic 을 쓴다. 패키지 루트는 이 모듈을 import 하지 않으므로
쓰는 쪽은 `from playground_llm.structured import generate_json` 으로 가져온다.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from playground_llm.errors import StructuredOutputError
from playground_llm.protocols import LLM
from playground_llm.types import Generation, Message, ModelSpec

RETRIES = 1  # 형식이 틀리면 오류를 보여 주고 한 번 더. 서버가 format 을 지키면 거의 쓰이지 않는다
MAX_ERRORS_SHOWN = 5  # 다시 요청할 때 보여 주는 검증 오류 수
RETRY_PROMPT = "앞 출력이 요구한 JSON 형식에 맞지 않는다. 오류:\n{errors}\n설명 없이 형식에 맞는 JSON 만 다시 출력한다."


@dataclass(frozen=True)
class JsonGeneration[T: BaseModel]:
    """검증한 값과 그 값을 낸 마지막 생성 결과."""

    value: T
    generation: Generation
    """마지막 시도의 생성 결과 (text 는 JSON 원문, usage 는 마지막 시도만)."""
    attempts: int
    """서버에 보낸 요청 수. 1 이면 첫 시도에 맞았다."""


def generate_json[T: BaseModel](
    llm: LLM,
    spec: ModelSpec,
    messages: Sequence[Message],
    schema: type[T],
    *,
    retries: int = RETRIES,
) -> JsonGeneration[T]:
    """주어진 schema 의 JSON Schema 를 서버에 요구하고, 답을 schema 로 검증해 돌려준다.

    형식이 틀리면 틀린 출력과 검증 오류를 대화에 붙여 retries 번까지 다시 요청한다.
    출력이 max_tokens 에 닿아 잘렸으면 다시 요청하지 않는다 — 같은 한도에서 또 잘린다.

    Args:
      llm: LLM 클라이언트.
      spec: 모델 · 생성 설정. think 를 켜도 된다 — thinking 은 따로 오고 text 만 검증한다.
      messages: 대화. 무엇을 JSON 으로 낼지는 프롬프트에도 적는 편이 결과가 좋다.
      schema: 출력 형식 pydantic 모델.
      retries: 형식이 틀렸을 때 다시 요청하는 최대 횟수 (0 이상).

    Raises:
      StructuredOutputError: 다시 요청해도 형식이 맞지 않을 때, 또는 잘려서 맞지 않을 때 (done_reason 이 length).
      GenerationError: LLM 호출이 실패했을 때 (generate 와 같은 예외들).
    """
    if retries < 0:
        raise ValueError(f"retries must be >= 0, got {retries=}")
    json_schema = schema.model_json_schema()
    conversation = list(messages)
    attempt = 1
    while True:
        generation = llm.generate(spec, conversation, json_schema=json_schema)
        try:
            value = schema.model_validate_json(generation.text)
        except ValidationError as e:
            if generation.done_reason == "length":
                raise StructuredOutputError(
                    f"{spec.name}: output was cut at max_tokens before a valid {schema.__name__}",
                    last_text=generation.text,
                    attempts=attempt,
                    done_reason="length",
                ) from e
            if attempt > retries:
                raise StructuredOutputError(
                    f"{spec.name}: output does not match {schema.__name__} after {attempt} attempts",
                    last_text=generation.text,
                    attempts=attempt,
                    done_reason=generation.done_reason,
                ) from e
            conversation += [generation.as_message(), Message("user", _retry_prompt(e))]
            attempt += 1
        else:
            return JsonGeneration(value=value, generation=generation, attempts=attempt)


def _retry_prompt(error: ValidationError) -> str:
    lines = [
        f"- {'.'.join(str(part) for part in item['loc']) or '(전체)'}: {item['msg']}"
        for item in error.errors()[:MAX_ERRORS_SHOWN]
    ]
    return RETRY_PROMPT.format(errors="\n".join(lines))
