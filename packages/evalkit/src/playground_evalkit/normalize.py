"""제목 · 정답 문자열 정규화. 평가하는 쪽(임베딩 벤치 · RAG 평가)이 같은 규칙을 쓰도록 한 곳에 둔다."""

import re
import unicodedata

_SPACES = re.compile(r"\s+")


def normalize_title(title: str) -> str:
    """문서 제목을 비교용으로 바꾼다: NFC · 밑줄 → 공백 · 연속 공백 하나로 · 앞뒤 공백 제거."""
    return _SPACES.sub(" ", unicodedata.normalize("NFC", title).replace("_", " ")).strip()


def squash(text: str) -> str:
    """정답 포함 여부 비교용: NFC 후 공백을 모두 지운다.

    위키 문서가 "증강현실" → "증강 현실" 처럼 띄어쓰기만 바뀌어도 정답을 찾도록 공백을 무시한다.
    """
    return _SPACES.sub("", unicodedata.normalize("NFC", text))
