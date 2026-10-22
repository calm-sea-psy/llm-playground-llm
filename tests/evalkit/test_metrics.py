import pytest

from playground_evalkit import hit_at_k, mrr, reciprocal_rank

RANKED = [
    [False, True, False],  # 2위
    [True, False, False],  # 1위
    [False, False, False],  # 없음
    [False, False, True],  # 3위
]


def test_hit_at_k_counts_queries_with_relevant_in_top_k():
    assert hit_at_k(RANKED, 1) == 0.25
    assert hit_at_k(RANKED, 2) == 0.5
    assert hit_at_k(RANKED, 3) == 0.75


def test_reciprocal_rank_and_mrr():
    assert [reciprocal_rank(row) for row in RANKED] == [0.5, 1.0, 0.0, pytest.approx(1 / 3)]
    assert mrr(RANKED) == pytest.approx((0.5 + 1.0 + 0.0 + 1 / 3) / 4)


def test_metrics_return_zero_for_no_queries():
    assert hit_at_k([], 5) == 0.0
    assert mrr([]) == 0.0


def test_hit_at_k_rejects_non_positive_k():
    with pytest.raises(ValueError, match="k must be"):
        hit_at_k(RANKED, 0)
