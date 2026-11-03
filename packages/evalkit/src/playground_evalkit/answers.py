"""답변 지표: KorQuAD 1.0 공식 EM · F1.

EM · F1 은 KorQuAD 1.0 공식 평가 스크립트(evaluate-v1.0.py)와 같은 규칙이다 — 다른 KorQuAD 결과와 비교할 수 있게.
  - 정규화: 따옴표 · 《》 · <> · 〈〉 · () · ‘’ 를 공백으로, 소문자, ASCII 구두점 제거, 연속 공백 하나로
  - EM: 정규화한 문자열이 같으면 1 (공백은 남긴다 — "증강현실" 과 "증강 현실" 은 다르다)
  - F1: 공백을 뺀 **문자** 단위 겹침의 조화 평균 (한국어라 단어 대신 문자)
  - 정답이 여럿이면 가장 높은 점수
"""

import re
import string
from collections import Counter
from collections.abc import Sequence

_QUOTES_AND_BRACKETS = re.compile("['\"《》<>〈〉()‘’]")
_PUNCTUATION = frozenset(string.punctuation)


def normalize_answer(text: str) -> str:
    """KorQuAD 1.0 공식 정규화."""
    text = _QUOTES_AND_BRACKETS.sub(" ", text).lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    return " ".join(text.split())


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    """정규화한 예측이 정답 중 하나와 같으면 1, 아니면 0. 정답이 없으면 0."""
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(answer) for answer in answers))


def f1(prediction: str, answers: Sequence[str]) -> float:
    """정답마다 문자 단위 F1 을 내고 가장 높은 값 (0~1). 정답이 없으면 0."""
    return max((_char_f1(prediction, answer) for answer in answers), default=0.0)


def _char_f1(prediction: str, answer: str) -> float:
    predicted = Counter(normalize_answer(prediction).replace(" ", ""))
    expected = Counter(normalize_answer(answer).replace(" ", ""))
    same = sum((predicted & expected).values())
    if same == 0:
        return 0.0
    precision = same / sum(predicted.values())
    recall = same / sum(expected.values())
    return 2 * precision * recall / (precision + recall)
