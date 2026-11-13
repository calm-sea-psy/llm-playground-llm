"""멀티턴 검색어: 바로 앞에서 되물은 질문을 이번 질문 앞에 붙여 검색한다.

되묻기(빠진 조건을 물은 답) 다음 질문은 그 조건에 대한 짧은 답일 때가 많아("1920년대요") 그것만으로는 검색되지 않는다.
되묻기가 아닌 턴을 만나면 거기서 멈춘다 — 그 앞은 이미 답이 나온 다른 질문이다.
되묻지 않은 후속 질문("그럼 그 다음 왕은?")은 이번 질문 그대로 검색한다 (질문을 고쳐 쓰지 않는다 — 알려진 한계).
"""

from collections.abc import Sequence

from playground_llm.context.slots import Turn

QUERY_SEPARATOR = " "
"""검색어에서 앞 질문 · 이번 질문 사이."""


def clarification_questions(turns: Sequence[Turn]) -> list[str]:
    """바로 앞에서 연달아 되묻기를 받은 질문들을 대화 순서대로 돌려준다.

    Args:
      turns: 앞선 턴, 대화 순서대로 (오래된 것 → 최신). 볼 최근 턴만 넘긴다.

    Returns:
      끝에서부터 `Turn.clarification` 이 이어진 턴의 질문. 직전 턴이 되묻기가 아니면 빈 목록.
    """
    picked: list[str] = []
    for turn in reversed(turns):
        if not turn.clarification:
            break
        picked.append(turn.question)
    picked.reverse()
    return picked


def search_query(question: str, prior_questions: Sequence[str], *, max_chars: int) -> str:
    """앞 질문들 뒤에 이번 질문을 이은 검색어를 돌려준다. 앞 질문이 없으면 이번 질문 그대로.

    Args:
      question: 이번 질문. 길어도 자르지 않고 늘 넣는다.
      prior_questions: 붙일 앞 질문, 대화 순서대로 (`clarification_questions` 의 결과).
      max_chars: 검색어 글자 상한 (임베더 · 리랭커 입력 한도). 넘치면 오래된 질문부터 빼고,
        앞 질문을 중간에서 자르지 않는다.
    """
    if max_chars < 1:
        raise ValueError(f"max_chars must be >= 1, got {max_chars=}")
    room = max_chars - len(question)
    kept: list[str] = []
    for prior in reversed(prior_questions):
        cost = len(prior) + len(QUERY_SEPARATOR)
        if cost > room:
            break  # 더 오래된 질문으로 건너뛰지 않는다 — 대화 순서가 끊기지 않게
        kept.append(prior)
        room -= cost
    kept.reverse()
    return QUERY_SEPARATOR.join([*kept, question])
