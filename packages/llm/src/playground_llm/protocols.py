"""LLM 클라이언트 계약. 서버마다 구현체(OllamaLLM · OpenAICompatLLM · FakeLLM)를 둔다."""

from collections.abc import Sequence
from typing import Protocol

from playground_llm.types import Generation, Message, ModelSpec


class LLM(Protocol):
    """메시지를 받아 생성 결과를 돌려주는 클라이언트.

    클라이언트 하나가 여러 설정을 받는다 — 답변 정책 단계마다 다른 ModelSpec(think · max_tokens)을 쓴다.
    서버를 바꿀 때는 구현체만 바꾸고 쓰는 쪽 코드는 그대로 둔다.
    """

    def generate(self, spec: ModelSpec, messages: Sequence[Message]) -> Generation:
        """주어진 spec 설정으로 messages 에 이어질 assistant 답을 만든다.

        Raises:
          ModelNotFoundError: 서버에 spec.name 모델이 없을 때.
          GenerationTimeoutError: 시간 제한을 넘겼을 때.
          GenerationError: 그 밖에 서버가 실패했을 때.
        """
        ...
