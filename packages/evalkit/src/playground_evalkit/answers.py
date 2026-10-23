"""답변 지표: KorQuAD 1.0 공식 EM · F1, 거부 정확도.

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
from dataclasses import dataclass

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


@dataclass(frozen=True)
class RefusalScores:
    """거부 판정 지표. '거부해야 하는 질문'(위키에 답이 없는 질문)을 양성으로 본다.

    분모가 0 인 값(예: 한 번도 거부하지 않았을 때 precision)은 0 으로 둔다 — 개수 필드로 구분한다.
    """

    precision: float
    """거부한 것 중 거부해야 했던 비율. 낮으면 답할 수 있는 질문을 거부했다."""
    recall: float
    """거부해야 하는 것 중 실제로 거부한 비율. 낮으면 답이 없는데 지어냈다."""
    f1: float
    accuracy: float
    """거부 여부를 맞힌 비율 (답할 질문은 답하고, 거부할 질문은 거부)."""
    false_refusal_rate: float
    """답할 수 있는 질문 중 거부한 비율 (= 1 − 답할 질문의 응답률)."""
    should_refuse: int
    """거부해야 하는 질문 수."""
    refused: int
    """실제로 거부한 질문 수."""


def refusal_scores(should_refuse: Sequence[bool], refused: Sequence[bool]) -> RefusalScores:
    """질문마다 (거부해야 하나, 실제로 거부했나) 로 거부 지표를 낸다.

    Raises:
      ValueError: 두 목록의 길이가 다를 때.
    """
    if len(should_refuse) != len(refused):
        raise ValueError(
            f"should_refuse and refused must have the same length, "
            f"got {len(should_refuse)=} {len(refused)=}"
        )
    pairs = list(zip(should_refuse, refused, strict=True))
    true_pos = sum(expected and actual for expected, actual in pairs)
    false_pos = sum(not expected and actual for expected, actual in pairs)
    positives = sum(should_refuse)
    negatives = len(pairs) - positives
    predicted = true_pos + false_pos
    precision = _ratio(true_pos, predicted)
    recall = _ratio(true_pos, positives)
    return RefusalScores(
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall),
        accuracy=_ratio(sum(expected == actual for expected, actual in pairs), len(pairs)),
        false_refusal_rate=_ratio(false_pos, negatives),
        should_refuse=positives,
        refused=predicted,
    )


def _char_f1(prediction: str, answer: str) -> float:
    predicted = Counter(normalize_answer(prediction).replace(" ", ""))
    expected = Counter(normalize_answer(answer).replace(" ", ""))
    same = sum((predicted & expected).values())
    if same == 0:
        return 0.0
    precision = same / sum(predicted.values())
    recall = same / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
