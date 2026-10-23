"""테스트용 LLM. 서버 없이 정해 둔 답을 돌려주고 받은 호출을 기록한다."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from playground_llm.types import Generation, Message, ModelSpec, Usage

type Reply = str | Callable[[ModelSpec, Sequence[Message]], str]
"""고정 답, 또는 (설정, 메시지) 를 보고 답을 만드는 함수."""


@dataclass(frozen=True)
class FakeCall:
    """FakeLLM 이 받은 호출 하나."""

    spec: ModelSpec
    messages: tuple[Message, ...]


class FakeLLM:
    """LLM Protocol 을 채우는 가짜 클라이언트. 답은 순서대로 쓰고, 마지막 답은 계속 반복한다.

    think 가 켜진 spec 이면 thinking 에 고정 문구를 넣어 thinking/text 분리를 시험할 수 있다.
    토큰 수는 공백으로 나눈 단어 수로 셈한다.

    예)
        llm = FakeLLM(["서울입니다."])
        llm.generate(spec, [Message("user", "조선의 수도는?")]).text  # "서울입니다."
        llm.calls[0].spec  # 받은 설정
    """

    THINKING = "(fake thinking)"

    def __init__(self, replies: Sequence[Reply]) -> None:
        if not replies:
            raise ValueError("replies must not be empty")
        self._replies = list(replies)
        self.calls: list[FakeCall] = []

    def generate(self, spec: ModelSpec, messages: Sequence[Message]) -> Generation:
        """다음 답을 돌려주고 호출을 calls 에 남긴다."""
        if not messages:
            raise ValueError("messages must not be empty")
        reply = self._replies[min(len(self.calls), len(self._replies) - 1)]
        self.calls.append(FakeCall(spec=spec, messages=tuple(messages)))
        text = reply(spec, messages) if callable(reply) else reply
        thinking = self.THINKING if spec.think else None
        completion = len(text.split()) + (len(thinking.split()) if thinking else 0)
        usage = Usage(
            prompt_tokens=sum(len(m.content.split()) for m in messages),
            completion_tokens=completion,
            total_ms=0,
        )
        return Generation(text=text, model=spec, usage=usage, thinking=thinking)
