"""검색 지표. 입력은 질의마다 순위대로 늘어놓은 정답 여부(bool) 목록이다."""

from collections.abc import Sequence


def hit_at_k(ranked_relevance: Sequence[Sequence[bool]], k: int) -> float:
    """상위 k 안에 정답이 하나라도 있는 질의 비율 (0~1). 질의가 없으면 0."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k=}")
    if not ranked_relevance:
        return 0.0
    return sum(any(row[:k]) for row in ranked_relevance) / len(ranked_relevance)


def reciprocal_rank(row: Sequence[bool]) -> float:
    """첫 정답 순위의 역수. 정답이 없으면 0."""
    return next((1.0 / rank for rank, relevant in enumerate(row, start=1) if relevant), 0.0)


def mrr(ranked_relevance: Sequence[Sequence[bool]]) -> float:
    """평균 역순위 (0~1). 질의가 없으면 0."""
    if not ranked_relevance:
        return 0.0
    return sum(reciprocal_rank(row) for row in ranked_relevance) / len(ranked_relevance)
