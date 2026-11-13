"""출처 범위 검사: 모델이 고른 출처 번호가 이번 호출의 번호 칸에 있는지 보고, 출처는 그 칸의 후보에서 채운다.

있는 문서여도 이번 호출의 문맥에 없었으면 모델이 근거로 쓸 수 없었다 — 앞 호출의 번호표를 다시 쓰면 엉뚱한 문서에
출처가 붙는다. 그래서 이번 `AssembledContext` 의 칸 결과만 기준으로 삼고, 제목 · 링크 같은 출처 정보는
모델이 쓴 글이 아니라 넣은 후보(`Placed.candidate`)에서 꺼낸다.
문장이 그 자료로 뒷받침되는지는 이 검사가 보지 않는다 (답변 평가의 몫).
"""

from collections.abc import Iterable, Sequence

from playground_llm.context.assembler import Placed, SlotResult
from playground_llm.errors import LLMError


class UnknownSourceError(LLMError, ValueError):
    """모델이 이번 문맥에 없는 출처 번호를 썼다. 메시지에는 번호만 넣는다 (모델 원문은 넣지 않는다)."""

    def __init__(self, unknown: Sequence[int]) -> None:
        super().__init__(f"source numbers not in this context: {list(unknown)}")
        self.unknown = tuple(unknown)
        """문맥에 없는 번호, 모델이 쓴 순서대로."""


def unknown_sources(numbers: Iterable[int], result: SlotResult) -> list[int]:
    """`result` 칸에 놓이지 않은 번호를 모델이 쓴 순서대로 돌려준다. 비어 있으면 모두 문맥 안이다.

    Args:
      numbers: 모델이 출처로 고른 번호 (예: 구조화 출력의 source_ids).
      result: 이번 호출의 번호 칸 결과 (`context.slot("documents")`). 번호 칸(`Slot.numbered`)이어야 한다.
    """
    by_number = _numbered(result)
    return [number for number in numbers if number not in by_number]


def cited_sources(numbers: Sequence[int], result: SlotResult) -> tuple[Placed, ...]:
    """모델이 고른 번호를 이번 호출에 넣은 후보로 바꾼 것을 돌려준다 — 모델이 쓴 순서, 중복 없이.

    Args:
      numbers: 모델이 출처로 고른 번호 (예: 구조화 출력의 source_ids).
      result: 이번 호출의 번호 칸 결과 (`context.slot("documents")`). 번호 칸(`Slot.numbered`)이어야 한다.

    Raises:
      UnknownSourceError: 칸에 없는 번호가 하나라도 있을 때. 일부만 맞아도 통과시키지 않는다.
    """
    by_number = _numbered(result)
    if unknown := [number for number in numbers if number not in by_number]:
        raise UnknownSourceError(unknown)
    return tuple(by_number[number] for number in dict.fromkeys(numbers))


def _numbered(result: SlotResult) -> dict[int, Placed]:
    if not result.slot.numbered:
        raise ValueError(f"{result.slot.name}: sources come from a numbered slot")
    return {placed.number: placed for placed in result.placed if placed.number is not None}
