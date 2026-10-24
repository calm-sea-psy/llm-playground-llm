import pytest

from playground_evalkit import mcnemar_exact, paired_bootstrap, wilson_interval


def _paired(
    only_a: int, only_b: int, both: int = 0, neither: int = 0
) -> tuple[list[bool], list[bool]]:
    correct_a = [True] * only_a + [False] * only_b + [True] * both + [False] * neither
    correct_b = [False] * only_a + [True] * only_b + [True] * both + [False] * neither
    return correct_a, correct_b


@pytest.mark.parametrize(
    ("successes", "total", "expected"),
    [
        # Newcombe (1998) 표 1 의 score 구간 (연속성 보정 없음) — 소수 넷째 자리
        (81, 263, (0.2553, 0.3662)),
        (15, 148, (0.0624, 0.1605)),
        (0, 20, (0.0, 0.1611)),
        (1, 29, (0.0061, 0.1718)),
        (29, 29, (0.8830, 1.0)),
        (10, 100, (0.0552, 0.1744)),
    ],
)
def test_wilson_interval_matches_published_values(successes, total, expected):
    assert wilson_interval(successes, total) == pytest.approx(expected, abs=1e-4)


def test_wilson_interval_upper_bound_for_zero_successes():
    low, high = wilson_interval(0, 30)

    assert low == 0.0  # 끝 경우는 부동소수 오차 없이 정확히 0
    assert high == pytest.approx(0.1135, abs=1e-4)  # 30개 중 0건이어도 실제 비율은 11% 까지 가능


def test_wilson_interval_for_all_successes_mirrors_zero_successes():
    low, high = wilson_interval(30, 30)

    assert high == 1.0
    assert low == pytest.approx(1 - wilson_interval(0, 30)[1])


def test_wilson_interval_widens_with_higher_confidence():
    low_95, high_95 = wilson_interval(10, 100)
    low_99, high_99 = wilson_interval(10, 100, confidence=0.99)

    assert low_99 < low_95
    assert high_99 > high_95
    assert (low_99, high_99) == pytest.approx((0.0460, 0.2038), abs=1e-4)


@pytest.mark.parametrize(
    ("successes", "total", "confidence", "match"),
    [
        (0, 0, 0.95, "total must be"),
        (31, 30, 0.95, "successes must be"),
        (-1, 30, 0.95, "successes must be"),
        (1, 30, 1.0, "confidence must be"),
        (1, 30, 0.0, "confidence must be"),
    ],
)
def test_wilson_interval_rejects_invalid_input(successes, total, confidence, match):
    with pytest.raises(ValueError, match=match):
        wilson_interval(successes, total, confidence=confidence)


def test_mcnemar_exact_matches_binomial_tail():
    # b=10 · c=2: 2 · P(X ≤ 2), X ~ Binomial(12, 1/2) = 2 · (1 + 12 + 66) / 4096 ≈ 0.0386
    result = mcnemar_exact(*_paired(10, 2, both=5, neither=3))

    assert (result.only_a, result.only_b) == (10, 2)
    assert result.p_value == pytest.approx(79 / 2048)


def test_mcnemar_exact_ignores_concordant_items():
    few = mcnemar_exact(*_paired(10, 2))
    many = mcnemar_exact(*_paired(10, 2, both=200, neither=100))

    assert many == few


def test_mcnemar_exact_is_symmetric():
    forward = mcnemar_exact(*_paired(10, 2))
    swapped = mcnemar_exact(*reversed(_paired(10, 2)))

    assert (swapped.only_a, swapped.only_b) == (2, 10)
    assert swapped.p_value == forward.p_value


def test_mcnemar_exact_for_large_counts():
    # 2 · P(X ≤ 40), X ~ Binomial(100, 1/2) — 카이제곱 근사(연속성 보정) 0.0574 에 가깝다
    assert mcnemar_exact(*_paired(60, 40)).p_value == pytest.approx(0.05689, abs=1e-5)


def test_mcnemar_exact_one_sided_split_cannot_reach_five_percent_with_five_items():
    assert mcnemar_exact(*_paired(5, 0)).p_value == pytest.approx(0.0625)  # 2 / 2**5


def test_mcnemar_exact_returns_one_for_equal_or_no_discordant():
    assert mcnemar_exact(*_paired(4, 4)).p_value == 1.0  # 2 · P(X ≤ 4) > 1 → 1
    no_split = mcnemar_exact(*_paired(0, 0, both=7, neither=3))
    assert no_split == mcnemar_exact([], [])
    assert (no_split.only_a, no_split.only_b, no_split.p_value) == (0, 0, 1.0)


def test_mcnemar_exact_rejects_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        mcnemar_exact([True, False], [True])


def test_paired_bootstrap_mean_difference_is_a_minus_b():
    result = paired_bootstrap([1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0])

    assert result.mean_difference == 0.5
    assert result.low <= result.mean_difference <= result.high


def test_paired_bootstrap_is_reproducible_with_same_seed():
    scores_a = [0.9, 0.4, 0.7, 1.0, 0.2, 0.6, 0.8, 0.5]
    scores_b = [0.8, 0.5, 0.4, 0.9, 0.2, 0.3, 0.9, 0.1]

    first = paired_bootstrap(scores_a, scores_b, seed=7)

    assert paired_bootstrap(scores_a, scores_b, seed=7) == first
    assert paired_bootstrap(scores_a, scores_b, seed=8) != first


def test_paired_bootstrap_is_close_to_normal_interval_for_large_sample():
    # 차이 1 이 60개 · 0 이 40개 → 평균 0.6, 표준오차 √(0.24/100) ≈ 0.049 → 정규 근사 (0.504, 0.696)
    result = paired_bootstrap([1.0] * 60 + [0.0] * 40, [0.0] * 100)

    assert result.mean_difference == pytest.approx(0.6)
    assert (result.low, result.high) == pytest.approx((0.504, 0.696), abs=0.015)


def test_paired_bootstrap_collapses_for_constant_difference():
    same = paired_bootstrap([0.3, 0.7, 0.5], [0.3, 0.7, 0.5])
    shifted = paired_bootstrap([0.5, 0.75, 1.0], [0.25, 0.5, 0.75])

    assert (same.mean_difference, same.low, same.high) == (0.0, 0.0, 0.0)
    assert (shifted.mean_difference, shifted.low, shifted.high) == (0.25, 0.25, 0.25)


def test_paired_bootstrap_narrows_with_lower_confidence():
    scores_a = [0.9, 0.4, 0.7, 1.0, 0.2, 0.6, 0.8, 0.5]
    scores_b = [0.8, 0.5, 0.4, 0.9, 0.2, 0.3, 0.9, 0.1]

    wide = paired_bootstrap(scores_a, scores_b)
    narrow = paired_bootstrap(scores_a, scores_b, confidence=0.8)

    assert wide.low <= narrow.low <= narrow.high <= wide.high


@pytest.mark.parametrize(
    ("scores_a", "scores_b", "options", "match"),
    [
        ([1.0, 0.0], [1.0], {}, "same length"),
        ([], [], {}, "must not be empty"),
        ([1.0], [0.0], {"resamples": 0}, "resamples must be"),
        ([1.0], [0.0], {"confidence": 1.5}, "confidence must be"),
    ],
)
def test_paired_bootstrap_rejects_invalid_input(scores_a, scores_b, options, match):
    with pytest.raises(ValueError, match=match):
        paired_bootstrap(scores_a, scores_b, **options)
