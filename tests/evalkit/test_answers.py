import pytest

from playground_evalkit import exact_match, f1, normalize_answer, percentile, refusal_scores


def test_normalize_follows_korquad_official_rules():
    assert normalize_answer("《세종대왕》") == "세종대왕"
    assert normalize_answer("'한양' (서울)") == "한양 서울"
    assert normalize_answer("Seoul,  Korea!") == "seoul korea"
    assert normalize_answer("「세종」") == "「세종」"  # 공식 규칙은 「」 를 지우지 않는다


def test_exact_match_keeps_spaces_but_ignores_quotes_and_case():
    assert exact_match("《한양》", ["한양"]) == 1.0
    assert exact_match("HanYang", ["hanyang"]) == 1.0
    assert exact_match("증강현실", ["증강 현실"]) == 0.0  # EM 은 띄어쓰기를 구분한다
    assert exact_match("한양", ["서울", "한양"]) == 1.0  # 정답 중 하나
    assert exact_match("한양", []) == 0.0


def test_f1_counts_characters_without_spaces():
    assert f1("증강현실", ["증강 현실"]) == 1.0  # F1 은 공백을 뺀 문자로 센다
    # 예측 "정답은한양" 5자 · 정답 "한양" 2자 · 겹침 2자 → P 0.4, R 1.0 → 2·0.4·1/1.4
    assert f1("정답은 한양", ["한양"]) == pytest.approx(0.8 / 1.4)
    assert f1("부산", ["한양"]) == 0.0
    assert f1("한", ["서울", "한양"]) == pytest.approx(2 / 3)  # 가장 높은 정답 기준
    assert f1("한양", []) == 0.0


def test_refusal_scores_treat_should_refuse_as_positive():
    should = [True, True, True, False, False, False, False]
    refused = [True, True, False, True, False, False, False]

    scores = refusal_scores(should, refused)

    assert scores.precision == pytest.approx(2 / 3)  # 거부 3번 중 맞은 2번
    assert scores.recall == pytest.approx(2 / 3)  # 거부해야 할 3개 중 2개
    assert scores.f1 == pytest.approx(2 / 3)
    assert scores.accuracy == pytest.approx(5 / 7)
    assert scores.false_refusal_rate == pytest.approx(1 / 4)  # 답할 4개 중 1개를 거부
    assert (scores.should_refuse, scores.refused) == (3, 3)


def test_refusal_scores_with_no_refusals_are_zero_not_errors():
    scores = refusal_scores([True, False], [False, False])

    assert (scores.precision, scores.recall, scores.f1) == (0.0, 0.0, 0.0)
    assert scores.refused == 0
    assert scores.accuracy == 0.5


def test_refusal_scores_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        refusal_scores([True], [])


def test_percentile_matches_linear_interpolation():
    values = [40.0, 10.0, 30.0, 20.0]

    assert percentile(values, 0) == 10.0
    assert percentile(values, 50) == 25.0
    assert percentile(values, 95) == pytest.approx(38.5)  # 위치 2.85 → 30 + 0.85·10
    assert percentile(values, 100) == 40.0
    assert percentile([7.0], 95) == 7.0


def test_percentile_rejects_bad_input():
    with pytest.raises(ValueError, match="empty"):
        percentile([], 50)
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0], 101)
