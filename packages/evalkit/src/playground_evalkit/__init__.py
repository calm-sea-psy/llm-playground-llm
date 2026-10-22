"""평가 지표 · KorQuAD 로더와 정답 매칭 · 결과 기록 · 리포트. 다른 패키지에 의존하지 않는다."""

from playground_evalkit.korquad import KorQuADItem, contains_answer, korquad_title, load_korquad
from playground_evalkit.metrics import hit_at_k, mrr, reciprocal_rank
from playground_evalkit.normalize import normalize_title, squash
from playground_evalkit.records import ParamValue, ResultRecord, read_records, write_records
from playground_evalkit.report import render_table

__all__ = [
    "KorQuADItem",
    "ParamValue",
    "ResultRecord",
    "contains_answer",
    "hit_at_k",
    "korquad_title",
    "load_korquad",
    "mrr",
    "normalize_title",
    "read_records",
    "reciprocal_rank",
    "render_table",
    "squash",
    "write_records",
]
