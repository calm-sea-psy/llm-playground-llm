import json
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType

from playground_llm import ModelSpec, Think

RUN_ID = "r1"
ORIGINAL_SET = "korquad-mini"
PARAPHRASE_SET = "paraphrase-mini"
COUNT = 30
NOTED_INDEX = 0  # 이 패러프레이즈에만 review_note 가 붙는다
RIGHT, WRONG = "서울", "부산"


def _item(item_id: str, kind: str, **extra: object) -> dict:
    return {
        "id": item_id,
        "kind": kind,
        "question": "한국의 수도는?",
        "answers": [RIGHT],
        "gold_title": "수도",
        "context": [{"rank": 1, "title": "수도", "text": "한국의 수도는 서울이다."}],
        **extra,
    }


def _paraphrase(index: int) -> dict:
    extra: dict[str, object] = {"source_id": f"korquad:{index}", "original_question": "원문"}
    if index == NOTED_INDEX:
        extra["review_note"] = "- 1번: 표현이 달라짐"
    return _item(f"paraphrase:{index}", "paraphrase", **extra)


def _output(model: str, mode: str, item_id: str, kind: str, *, correct: bool) -> dict:
    return {
        "spec_id": f"ollama/{model}?think=off",
        "model": model,
        "think": "off",
        "mode": mode,
        "item_id": item_id,
        "kind": kind,
        "prompt_version": "p@v1",
        "answer": RIGHT if correct else WRONG,
        "evidence": "수도",
        "attempts": 1,
        "error": None,
        "total_ms": 100,
        "done_reason": "stop",
    }


def _outputs(
    model: str,
    *,
    norag_original: Sequence[bool],
    norag_paraphrase: Sequence[bool] = (),
    rag_original: Sequence[bool] = (True,) * COUNT,
    rag_paraphrase: Sequence[bool] = (),
) -> list[dict]:
    """한 spec 의 출력. 각 줄은 문항 i 를 맞혔는지 — 비우면 그 (모드, 종류) 출력이 없다."""
    columns = (
        ("rag", "korquad", "answerable", rag_original),
        ("norag", "korquad", "answerable", norag_original),
        ("norag", "paraphrase", "paraphrase", norag_paraphrase),
        ("rag", "paraphrase", "paraphrase", rag_paraphrase),
    )
    rows = []
    for mode, prefix, kind, correct in columns:
        rows.extend(
            _output(model, mode, f"{prefix}:{index}", kind, correct=hit)
            for index, hit in enumerate(correct)
        )
    return rows


def _score(
    f4_bench: ModuleType, tmp_path: Path, outputs: Sequence[dict], *, paraphrase_set: bool = True
) -> str:
    """합성 세트 · 출력으로 score 를 돌리고 리포트 글을 돌려준다."""
    data = tmp_path / "data"
    sets = {ORIGINAL_SET: [_item(f"korquad:{index}", "answerable") for index in range(COUNT)]}
    if paraphrase_set:
        sets[PARAPHRASE_SET] = [_paraphrase(index) for index in range(COUNT)]
    for name, items in sets.items():
        _write_jsonl(data / name / "items.jsonl", items)
    run_dir = data / "runs" / RUN_ID
    run_dir.mkdir(parents=True)
    (run_dir / "meta.json").write_text(json.dumps({"sets": list(sets)}), encoding="utf-8")
    _write_jsonl(run_dir / "outputs.jsonl", outputs)
    reports = tmp_path / "reports"
    f4_bench.score(RUN_ID, data=data, reports=reports)
    return (reports / f"{RUN_ID}.md").read_text(encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def _table(report: str, column: str) -> list[dict[str, str]]:
    """리포트에서 이 칸 이름이 든 첫 markdown 표의 줄들."""
    lines = report.splitlines()
    for index, line in enumerate(lines):
        header = _cells(line)
        if line.startswith("| ") and column in header:
            rows = []
            for row_line in lines[index + 2 :]:
                if not row_line.startswith("| "):
                    break
                rows.append(dict(zip(header, _cells(row_line), strict=True)))
            return rows
    raise AssertionError(f"no table with column {column!r}")


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _row(rows: list[dict[str, str]], model: str, label: str) -> dict[str, str]:
    return next(row for row in rows if row["spec"].startswith(model) and row["문항"] == label)


def _top(report: str) -> str:
    """제목과 실행 설명(세트 줄) 사이 — 맨 위 알림이 들어가는 곳."""
    lines = report.splitlines()
    end = next(index for index, line in enumerate(lines) if line.startswith("세트 "))
    return "\n".join(lines[:end])


def test_score_fires_when_norag_paraphrase_gap_is_large(f4_bench: ModuleType, tmp_path: Path):
    report = _score(
        f4_bench,
        tmp_path,
        _outputs("memo", norag_original=[True] * COUNT, norag_paraphrase=[False] * COUNT),
    )

    row = _row(_table(report, "패러프레이즈 F1"), "memo", "전체")
    assert row["원문 F1"] == "100.0"
    assert row["패러프레이즈 F1"] == "0.0"
    assert row["차이"] == "100.0"
    assert row["95% 구간"] == "100.0 ~ 100.0"
    assert row["판정"] == "확인 단계 필요"
    assert "오염 판정 발동" in _top(report)
    assert "--specs memo --think off --paraphrase-rag" in _top(report)
    assert "확인 단계 출력 없음: `memo`" in report


def test_score_does_not_fire_when_gap_is_small(f4_bench: ModuleType, tmp_path: Path):
    same = [index % 3 != 0 for index in range(COUNT)]
    report = _score(
        f4_bench, tmp_path, _outputs("steady", norag_original=same, norag_paraphrase=same)
    )

    row = _row(_table(report, "패러프레이즈 F1"), "steady", "전체")
    assert row["차이"] == "0.0"
    assert row["판정"] == "-"
    assert "오염 판정 발동" not in report
    assert "발동한 spec 이 없어 건너뛴다." in report


def test_score_does_not_fire_when_lower_bound_is_not_above_zero(
    f4_bench: ModuleType, tmp_path: Path
):
    # 6문항은 원문만, 3문항은 패러프레이즈만 맞힘 → 평균 차이 10%p 지만 구간이 0 을 넘나든다
    original = [index < 6 for index in range(COUNT)]
    paraphrase = [6 <= index < 9 for index in range(COUNT)]
    report = _score(
        f4_bench, tmp_path, _outputs("noisy", norag_original=original, norag_paraphrase=paraphrase)
    )

    row = _row(_table(report, "패러프레이즈 F1"), "noisy", "전체")
    assert row["차이"] == "10.0"
    assert float(row["95% 구간"].split(" ~ ")[0]) <= 0
    assert row["판정"] == "-"
    assert "오염 판정 발동" not in report


def test_score_moves_ranking_when_memorization_estimate_is_large(
    f4_bench: ModuleType, tmp_path: Path
):
    outputs = _outputs(
        "memo",
        norag_original=[True] * COUNT,
        norag_paraphrase=[False] * COUNT,
        rag_paraphrase=[True] * COUNT,
    ) + _outputs("steady", norag_original=[True] * COUNT, norag_paraphrase=[True] * COUNT)
    report = _score(f4_bench, tmp_path, outputs)

    rows = _table(report, "외움 추정")
    row = _row(rows, "memo", "전체")
    assert row["n"] == str(COUNT)
    assert row["no-RAG 차이"] == "100.0"
    assert row["with-RAG 차이"] == "0.0"
    assert row["외움 추정"] == "100.0"
    assert row["판정"] == "롱테일 100 으로 옮김"
    assert not [row for row in rows if row["spec"].startswith("steady")]  # rag 출력이 없는 spec
    assert "1단계 순위 기준을 롱테일 100 으로 옮김" in _top(report)
    assert "확인 단계 출력 없음" not in report
    assert "가정 1:" in report
    assert "가정 2:" in report


def test_score_keeps_ranking_when_rag_gap_is_as_large(f4_bench: ModuleType, tmp_path: Path):
    # 문맥이 있어도 패러프레이즈를 못 맞힘 → 외움이 아니라 표현 민감도
    report = _score(
        f4_bench,
        tmp_path,
        _outputs(
            "fragile",
            norag_original=[True] * COUNT,
            norag_paraphrase=[False] * COUNT,
            rag_paraphrase=[False] * COUNT,
        ),
    )

    row = _row(_table(report, "외움 추정"), "fragile", "전체")
    assert row["with-RAG 차이"] == "100.0"
    assert row["외움 추정"] == "0.0"
    assert row["판정"] == "옮기지 않음"
    assert "오염 판정 발동" in _top(report)
    assert "롱테일 100 으로 옮김" not in _top(report)


def test_score_adds_row_without_review_note_items(f4_bench: ModuleType, tmp_path: Path):
    # 검수 비고가 붙은 문항만 원문 · 패러프레이즈가 갈린다 → 빼면 차이가 0
    paraphrase = [index != NOTED_INDEX for index in range(COUNT)]
    report = _score(
        f4_bench,
        tmp_path,
        _outputs(
            "memo",
            norag_original=[True] * COUNT,
            norag_paraphrase=paraphrase,
            rag_paraphrase=paraphrase,
        ),
    )

    rows = _table(report, "패러프레이즈 F1")
    assert [row["문항"] for row in rows] == ["전체", "검수 비고 뺌"]
    whole, kept = rows
    assert whole["n"] == str(COUNT)
    assert whole["차이"] == f"{100 / COUNT:.1f}"
    assert kept["n"] == str(COUNT - 1)
    assert kept["차이"] == "0.0"
    confirm = _table(report, "외움 추정")
    assert [row["문항"] for row in confirm] == ["전체", "검수 비고 뺌"]
    assert confirm[1]["n"] == str(COUNT - 1)
    assert all(row["판정"] == "참고 (발동 안 한 spec)" for row in confirm)


def test_score_reports_missing_paraphrase_outputs_and_keeps_tables(
    f4_bench: ModuleType, tmp_path: Path
):
    outputs = _outputs("memo", norag_original=[True] * COUNT)
    without_set = _score(f4_bench, tmp_path / "a", outputs, paraphrase_set=False)
    no_outputs = _score(f4_bench, tmp_path / "b", outputs)
    # 발동하지 않는 패러프레이즈 출력 — 맨 위 알림 없이 기존 표만 비교한다
    with_outputs = _score(
        f4_bench,
        tmp_path / "c",
        _outputs("memo", norag_original=[True] * COUNT, norag_paraphrase=[True] * COUNT),
    )

    assert "오염 판정" not in without_set
    assert "패러프레이즈 출력 없음" in no_outputs
    assert "패러프레이즈 출력 없음" not in with_outputs
    # 세트 이름 줄만 빼면 기존 표와 설명은 그대로이고 오염 판정 절만 뒤에 붙는다
    before = _without_set_line(without_set)
    for report in (no_outputs, with_outputs):
        after = _without_set_line(report)
        assert after[: len(before)] == before
        assert after[len(before) : len(before) + 2] == ["", "## 오염 판정 (원문 vs 패러프레이즈)"]


def _without_set_line(report: str) -> list[str]:
    return [line for line in report.splitlines() if not line.startswith("세트 ")]


def test_score_records_paraphrase_scores_like_answerable(f4_bench: ModuleType, tmp_path: Path):
    _score(
        f4_bench,
        tmp_path,
        _outputs("memo", norag_original=[True] * COUNT, norag_paraphrase=[True] * COUNT),
    )

    lines = (tmp_path / "data" / "runs" / RUN_ID / "records.jsonl").read_text(encoding="utf-8")
    records = [json.loads(line) for line in lines.splitlines()]
    paraphrase = next(r for r in records if r["item_id"] == "norag/paraphrase:1")
    assert paraphrase["scores"]["em"] == 1.0
    assert paraphrase["scores"]["f1"] == 1.0
    assert paraphrase["scores"]["contains"] == 1.0


def _spec(think: Think) -> ModelSpec:
    return ModelSpec("ollama", "gemma4:12b", {"think": think})


def _picked(f4_bench: ModuleType, spec: ModelSpec, *, paraphrase_rag: bool) -> set[tuple]:
    items = [_item("korquad:0", "answerable"), _paraphrase(0)]
    jobs = f4_bench._jobs(spec, items, set(), paraphrase_rag=paraphrase_rag)
    return {(mode, item["id"]) for mode, item in jobs}


def test_jobs_runs_paraphrase_without_context_by_default(f4_bench: ModuleType):
    picked = _picked(f4_bench, _spec(Think.OFF), paraphrase_rag=False)

    assert picked == {
        ("rag", "korquad:0"),
        ("norag", "korquad:0"),
        ("norag", "paraphrase:0"),
    }


def test_jobs_adds_paraphrase_rag_with_flag(f4_bench: ModuleType):
    picked = _picked(f4_bench, _spec(Think.OFF), paraphrase_rag=True)

    assert ("rag", "paraphrase:0") in picked
    assert ("norag", "paraphrase:0") in picked


def test_jobs_skips_paraphrase_for_spec_without_norag(f4_bench: ModuleType):
    # --specs gemma4:12b 는 think ON spec 도 고른다 — 짝지을 no-RAG 가 없으니 패러프레이즈를 돌지 않는다
    picked = _picked(f4_bench, _spec(Think.ON), paraphrase_rag=True)

    assert picked == {("rag", "korquad:0")}


def test_jobs_skips_done(f4_bench: ModuleType):
    spec = _spec(Think.OFF)
    done = {(spec.spec_id, "norag", "paraphrase:0")}
    jobs = f4_bench._jobs(spec, [_paraphrase(0)], done, paraphrase_rag=True)

    assert [(mode, item["id"]) for mode, item in jobs] == [("rag", "paraphrase:0")]
