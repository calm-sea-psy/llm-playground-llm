"""검색 지표 (입력은 질의마다 순위대로 늘어놓은 정답 여부 목록) 와 지연 백분위수."""

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


def percentile(values: Sequence[float], q: float) -> float:
    """Q 백분위수 (0~100). 정렬한 값 사이를 선형 보간한다 (numpy.percentile 기본값과 같다).

    Raises:
      ValueError: values 가 비었거나 q 가 0~100 밖일 때.
    """
    if not values:
        raise ValueError("values must not be empty")
    if not 0 <= q <= 100:  # noqa: PLR2004  # 백분위 범위
        raise ValueError(f"q must be in [0, 100], got {q=}")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
