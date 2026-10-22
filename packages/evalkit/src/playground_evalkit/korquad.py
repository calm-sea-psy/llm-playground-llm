"""KorQuAD 1.0 로더와 "문서 제목 + 정답 문자열 포함" 매칭."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from playground_evalkit.normalize import normalize_title, squash

KORQUAD_COLON = "__"  # KorQuAD 제목은 쌍점을 밑줄 두 개로 적는다 (커맨드_앤_컨커_3__타이베리움_워)


@dataclass(frozen=True)
class KorQuADItem:
    """KorQuAD 질문 하나."""

    id: str
    """KorQuAD 질문 id."""
    title: str
    """정규화하고 title_aliases 로 보정한 문서 제목 — 코퍼스 문서 제목과 비교하는 값."""
    question: str
    answer: str
    """첫 번째 정답 문자열."""
    source_title: str
    """KorQuAD 원래 제목 (밑줄 표기)."""


def korquad_title(title: str) -> str:
    """KorQuAD 제목을 코퍼스 제목과 같은 모양으로: `__` → 쌍점, 그다음 normalize_title."""
    return normalize_title(title.replace(KORQUAD_COLON, ": "))


def load_korquad(
    path: Path, *, title_aliases: Mapping[str, str] | None = None
) -> list[KorQuADItem]:
    """KorQuAD 1.0 JSON 을 질문 목록으로 읽는다.

    Args:
      path: KorQuAD 1.0 JSON (train 또는 dev).
      title_aliases: 옛 제목 → 현재 제목. 넘겨주기 목록이나 서브셋 매칭표(`korquad_title_map`)를 그대로 넘긴다.
        키 · 값 양쪽을 같은 규칙으로 정규화한 뒤 쓴다.

    Returns:
      파일 순서대로의 질문.
    """
    aliases = {
        korquad_title(old): normalize_title(new) for old, new in (title_aliases or {}).items()
    }
    data = json.loads(path.read_text(encoding="utf-8"))["data"]
    items: list[KorQuADItem] = []
    for article in data:
        title = korquad_title(article["title"])
        title = aliases.get(title, title)
        for paragraph in article["paragraphs"]:
            items.extend(
                KorQuADItem(
                    id=qa["id"],
                    title=title,
                    question=qa["question"],
                    answer=qa["answers"][0]["text"],
                    source_title=article["title"],
                )
                for qa in paragraph["qas"]
            )
    return items


def contains_answer(title: str, text: str, item: KorQuADItem) -> bool:
    """청크가 질문의 정답 근거인지: 문서 제목이 같고, 공백을 무시했을 때 정답 문자열이 본문에 있다."""
    return normalize_title(title) == item.title and squash(item.answer) in squash(text)
