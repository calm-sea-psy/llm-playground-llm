"""평가 지표(검색 · 답변 EM/F1 · 거부) · 벤치 비교 통계 · KorQuAD 로더와 정답 매칭 · 결과 기록 · 리포트. 다른 패키지에 의존하지 않는다."""

from playground_evalkit.answers import (
    RefusalScores,
    exact_match,
    f1,
    normalize_answer,
    refusal_scores,
)
from playground_evalkit.korquad import KorQuADItem, contains_answer, korquad_title, load_korquad
from playground_evalkit.metrics import hit_at_k, mrr, percentile, reciprocal_rank
from playground_evalkit.normalize import normalize_title, squash
from playground_evalkit.records import ParamValue, ResultRecord, read_records, write_records
from playground_evalkit.report import render_table
from playground_evalkit.stats import (
    McNemarResult,
    PairedBootstrap,
    mcnemar_exact,
    paired_bootstrap,
    wilson_interval,
)

__all__ = [
    "KorQuADItem",
    "McNemarResult",
    "PairedBootstrap",
    "ParamValue",
    "RefusalScores",
    "ResultRecord",
    "contains_answer",
    "exact_match",
    "f1",
    "hit_at_k",
    "korquad_title",
    "load_korquad",
    "mcnemar_exact",
    "mrr",
    "normalize_answer",
    "normalize_title",
    "paired_bootstrap",
    "percentile",
    "read_records",
    "reciprocal_rank",
    "refusal_scores",
    "render_table",
    "squash",
    "wilson_interval",
    "write_records",
]
