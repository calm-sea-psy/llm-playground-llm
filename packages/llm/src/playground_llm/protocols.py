"""LLM 클라이언트 계약. 서버마다 구현체(OllamaLLM · OpenAICompatLLM · FakeLLM)를 둔다."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Protocol

from playground_llm.types import Generation, Message, ModelSpec, StreamDelta


class LLM(Protocol):
    """메시지를 받아 생성 결과를 돌려주는 클라이언트.

    클라이언트 하나가 여러 설정을 받는다 — 답변 정책 단계마다 다른 ModelSpec(think · max_tokens)을 쓴다.
    서버를 바꿀 때는 구현체만 바꾸고 쓰는 쪽 코드는 그대로 둔다.
    """

    def generate(
        self,
        spec: ModelSpec,
        messages: Sequence[Message],
        *,
        json_schema: Mapping[str, object] | None = None,
    ) -> Generation:
        """주어진 spec 설정으로 messages 에 이어질 assistant 답을 만든다.

        Args:
          spec: 모델 · 생성 설정.
          messages: 대화. 이전 턴의 thinking 은 넣지 않는다.
          json_schema: 있으면 답(text)을 이 JSON Schema 에 맞춰 내도록 서버에 요구한다
            (Ollama `format`). 서버가 지키지 못할 수도 있어 검증은 쓰는 쪽(generate_json)이 한다.

        Raises:
          ModelNotFoundError: 서버에 spec.name 모델이 없을 때.
          GenerationTimeoutError: 시간 제한을 넘겼을 때.
          GenerationError: 그 밖에 서버가 실패했을 때.
        """
        ...

    def stream(
        self, spec: ModelSpec, messages: Sequence[Message]
    ) -> Iterator[StreamDelta | Generation]:
        """조각 단위로 받는 generate. 조각(StreamDelta)들 뒤에 마지막으로 전체 Generation 하나를 보낸다.

        Raises:
          generate 와 같다. 조각을 몇 개 보낸 뒤에도 날 수 있다.
        """
        ...
