"""벤치 비교 통계: 비율의 Wilson 구간, 정답 여부의 McNemar 정확 검정, 짝지은 점수의 bootstrap 구간."""

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from playground_evalkit.metrics import percentile

CONFIDENCE = 0.95  # 리포트에 적는 기본 신뢰 수준
# 백분위 bootstrap 으로 95% 구간 끝(2.5 · 97.5 백분위)을 낼 때 흔히 권하는 반복 수
BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True)
class McNemarResult:
    """같은 문항을 푼 두 모델 a · b 의 McNemar 정확 검정 결과."""

    only_a: int
    """a 만 맞힌 문항 수 (교과서 표기 b)."""
    only_b: int
    """b 만 맞힌 문항 수 (교과서 표기 c)."""
    p_value: float
    """양측 p 값 (0~1). 엇갈린 문항이 없으면 1."""


@dataclass(frozen=True)
class PairedBootstrap:
    """짝지은 점수 두 줄 a · b 의 평균 차이와 bootstrap 백분위 구간."""

    mean_difference: float
    """평균(a) − 평균(b). 양수면 a 가 높다."""
    low: float
    """구간 하한."""
    high: float
    """구간 상한."""


def wilson_interval(
    successes: int, total: int, *, confidence: float = CONFIDENCE
) -> tuple[float, float]:
    """비율 successes / total 의 Wilson 점수 구간 (하한, 상한) 을 돌려준다 (0~1).

    정규 근사 구간(p ± z·√(p(1−p)/n))과 달리 0건 · 전체 성공에서도 폭이 0 이 되지 않는다
    — 예: 0/30 이어도 95% 상한은 약 0.114.

    Raises:
      ValueError: total 이 1 미만이거나, successes 가 0~total 밖이거나, confidence 가 (0, 1) 밖일 때.
    """
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total=}")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, total], got {successes=} {total=}")
    _check_confidence(confidence)
    z_score = NormalDist().inv_cdf((1 + confidence) / 2)
    rate = successes / total
    shrink = z_score**2 / total  # z²/n — 작은 표본일수록 중심을 1/2 쪽으로 당긴다
    center = (rate + shrink / 2) / (1 + shrink)
    half_width = (
        z_score / (1 + shrink) * math.sqrt(rate * (1 - rate) / total + shrink / (4 * total))
    )
    # 끝 경우는 식으로도 정확히 0 · 1 이지만, 부동소수 오차(±1e-17)가 남지 않게 값을 바로 둔다
    low = 0.0 if successes == 0 else center - half_width
    high = 1.0 if successes == total else center + half_width
    return low, high


def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """문항마다 (a 정답 여부, b 정답 여부) 로 McNemar 정확 검정(양측)을 한다.

    맞음/틀림이 엇갈린 문항만 본다. 두 모델 실력이 같다면 a 만 맞힌 수는 Binomial(엇갈린 수, 1/2) 을
    따르므로, p 값 = min(1, 2 · P(X ≤ 작은 쪽)). 이항 분포를 그대로 세므로 엇갈린 문항이 적어도
    (카이제곱 근사가 맞지 않는 25건 미만) 쓸 수 있다. 문항이 없으면 엇갈림 0 으로 본다.

    Raises:
      ValueError: 두 목록의 길이가 다를 때.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError(
            f"correct_a and correct_b must have the same length, "
            f"got {len(correct_a)=} {len(correct_b)=}"
        )
    pairs = list(zip(correct_a, correct_b, strict=True))
    only_a = sum(hit_a and not hit_b for hit_a, hit_b in pairs)
    only_b = sum(hit_b and not hit_a for hit_a, hit_b in pairs)
    return McNemarResult(
        only_a=only_a, only_b=only_b, p_value=_binomial_two_sided_p(only_a, only_b)
    )


def paired_bootstrap(
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    confidence: float = CONFIDENCE,
    seed: int = 0,
) -> PairedBootstrap:
    """문항마다 짝지은 점수 a · b 의 평균 차이(a − b)와 bootstrap 백분위 구간을 돌려준다.

    문항 단위로 복원 추출해 차이의 평균을 resamples 번 내고, 그 분포의 (1 − confidence)/2 ·
    (1 + confidence)/2 백분위를 구간으로 쓴다. 짝을 지키므로 문항 난이도 차이가 구간에 섞이지 않는다.
    같은 seed 면 같은 결과.

    Raises:
      ValueError: 두 목록의 길이가 다르거나 비었을 때, resamples 가 1 미만이거나 confidence 가
        (0, 1) 밖일 때.
    """
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"scores_a and scores_b must have the same length, "
            f"got {len(scores_a)=} {len(scores_b)=}"
        )
    if not scores_a:
        raise ValueError("scores must not be empty")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples=}")
    _check_confidence(confidence)
    differences = [score_a - score_b for score_a, score_b in zip(scores_a, scores_b, strict=True)]
    count = len(differences)
    rng = random.Random(seed)  # 전역 random 상태와 따로 — seed 만으로 결과가 정해진다
    means = [sum(rng.choices(differences, k=count)) / count for _ in range(resamples)]
    tail = (1 - confidence) / 2 * 100
    return PairedBootstrap(
        mean_difference=sum(differences) / count,
        low=percentile(means, tail),
        high=percentile(means, 100 - tail),
    )


def _binomial_two_sided_p(only_a: int, only_b: int) -> float:
    discordant = only_a + only_b
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, count) for count in range(min(only_a, only_b) + 1))
    # 정수로 세고 마지막에 한 번 나눈다 — 2 ** discordant 가 커도 int / int 는 정확히 반올림된다
    return min(1.0, tail / 2 ** (discordant - 1))


def _check_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence=}")
