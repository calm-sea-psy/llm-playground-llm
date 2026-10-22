from pathlib import Path

import pytest

from playground_evalkit import (
    ResultRecord,
    normalize_title,
    read_records,
    render_table,
    squash,
    write_records,
)


def test_records_round_trip_through_jsonl(tmp_path: Path):
    records = [
        ResultRecord(
            run_id="r1",
            spec_id="bge-m3/384",
            item_id="q1",
            scores={"hit@5": 1.0, "rr": 0.5},
            spec={"model": "bge-m3", "max_tokens": 384, "fp16": True, "stop": ("\n", "끝")},
            created_at="2026-10-23T09:00:00+09:00",
        ),
        ResultRecord(run_id="r1", spec_id="bge-m3/384", item_id="q2", scores={"hit@5": 0.0}),
    ]
    path = tmp_path / "out" / "records.jsonl"

    assert write_records(path, records) == 2
    assert read_records(path) == records


def test_render_table_formats_floats_and_blank_cells():
    table = render_table(
        [{"model": "bge-m3", "hit@5": 0.91234}, {"model": "koe5"}],
        ["model", "hit@5"],
    )

    assert table == "| model | hit@5 |\n|---|---|\n| bge-m3 | 0.912 |\n| koe5 |  |\n"


def test_render_table_needs_columns():
    with pytest.raises(ValueError, match="columns"):
        render_table([], [])


def test_normalize_title_and_squash():
    assert normalize_title("  알렉산더_헤이그 ") == "알렉산더 헤이그"
    assert squash("증강 현실\n기술") == "증강현실기술"
