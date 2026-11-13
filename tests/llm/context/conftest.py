from collections.abc import Callable

import pytest

from playground_llm.context import ContextAssembler, ContextLayout, Slot


class LenCounter:
    """글자 하나 = 토큰 하나. 경계 계산을 눈으로 확인하려고."""

    def count(self, text: str) -> int:
        return len(text)


@pytest.fixture
def make_assembler() -> Callable[..., ContextAssembler]:
    """칸들로 레이아웃을 만들어 글자 수로 세는(`LenCounter`) 조립기를 돌려주는 함수."""

    def make(
        *slots: Slot, budget: int = 10_000, reserved: frozenset[str] = frozenset()
    ) -> ContextAssembler:
        return ContextAssembler(ContextLayout(slots, budget, reserved), LenCounter())

    return make
