"""중복 정리 규칙: key 마다 점수가 가장 높은 하나만 남기고, 자리는 처음 나온 자리를 지킨다.

조립기의 후보 선별과 작업 기억의 근거 쌓기가 같은 규칙을 쓴다 — 한쪽만 바뀌면 같은 근거가 칸과 기억에서 다르게 남는다.
"""

from collections.abc import Callable, Iterable, Sequence
from typing import Protocol


class _Scored(Protocol):
    """점수로 견주는 항목."""

    @property
    def score(self) -> float:
        """높을수록 좋은 점수."""
        ...


def merge_best[ItemT: _Scored](
    found: Sequence[ItemT], new: Iterable[ItemT], *, key: Callable[[ItemT], str | None]
) -> tuple[list[ItemT], list[ItemT]]:
    """`found` 뒤에 `new` 를 붙이며 key 마다 점수가 가장 높은 하나만 남긴다.

    Args:
      found: 이미 정리된 목록. 그대로 앞에 둔다 (안의 중복은 다시 보지 않는다).
      new: 뒤에 붙일 항목, 받은 순서대로.
      key: 중복 판단 키. None 을 돌려주는 항목은 비교하지 않고 늘 붙인다.

    Returns:
      (남긴 항목, 뺀 항목). 남긴 항목은 처음 나온 자리에 더 높은 점수가 들어가고, 점수가 같으면 먼저 나온 것을 남긴다.
      뺀 항목은 뺀 순서대로.
    """
    merged = list(found)
    index = {item_key: i for i, item in enumerate(merged) if (item_key := key(item)) is not None}
    dropped: list[ItemT] = []
    for item in new:
        item_key = key(item)
        if item_key is None:
            merged.append(item)
            continue
        seen = index.get(item_key)
        if seen is None:
            index[item_key] = len(merged)
            merged.append(item)
        elif item.score > merged[seen].score:
            dropped.append(merged[seen])
            merged[seen] = item
        else:
            dropped.append(item)
    return merged, dropped


def keep_best[ItemT: _Scored](
    found: Sequence[ItemT], new: Iterable[ItemT], *, key: Callable[[ItemT], str]
) -> tuple[ItemT, ...]:
    """`found` 뒤에 `new` 를 붙인 목록 — key 마다 하나, 더 높은 점수를 남기고 자리는 처음 나온 자리. 점수가 같으면 먼저 나온 것."""
    merged, _ = merge_best(found, new, key=key)
    return tuple(merged)
