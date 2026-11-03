"""평가 결과 기록. 문항 id 별 점수만 남기고 원문(질문 · 정답 · 모델 출력)은 남기지 않는다."""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType

# playground_llm.ParamValue 와 같은 정의 — ResultRecord.spec 이 ModelSpec.params 를 그대로 받는다.
# 두 패키지는 서로 의존하지 않으므로 한 줄을 각자 둔다 (tests/llm/test_types.py 가 맞는지 확인한다)
type ParamValue = str | int | float | bool | tuple[str, ...]
"""설정 값. tuple 은 Ollama `stop` 같은 문자열 목록."""


@dataclass(frozen=True)
class ResultRecord:
    """한 실행 · 한 문항의 점수. 만든 뒤에는 scores · spec 도 고칠 수 없다.

    `dataclasses.asdict` 는 읽기 전용 매핑을 복사하지 못하므로 쓰지 않는다. 파일로는 `write_records` 로.
    """

    run_id: str
    """실행 이름 (예: `2026-10-23-embed-compare`)."""
    spec_id: str
    """평가 대상 식별자 (예: `bge-m3/384`)."""
    item_id: str
    """문항 id (KorQuAD 질문 id 등)."""
    scores: Mapping[str, float]
    """지표 이름 → 값 (예: `{"hit@5": 1.0, "rr": 0.5}`)."""
    spec: Mapping[str, ParamValue] = field(default_factory=dict)
    """평가 대상 설정 (모델 · 청크 크기 · 토크나이저 식별자 등)."""
    prompt_version: str | None = None
    """생성 평가일 때 프롬프트 버전. 검색 평가는 None."""
    created_at: str = ""
    """ISO 8601 시각."""

    def __post_init__(self) -> None:
        # 만든 쪽이 넘긴 dict 를 나중에 고쳐도 기록이 바뀌지 않게 읽기 전용 복사본으로
        object.__setattr__(self, "scores", MappingProxyType(dict(self.scores)))
        object.__setattr__(self, "spec", MappingProxyType(dict(self.spec)))


def write_records(path: Path, records: Iterable[ResultRecord]) -> int:
    """기록을 jsonl 로 쓰고 쓴 줄 수를 돌려준다. 상위 폴더가 없으면 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(_json_row(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_records(path: Path) -> list[ResultRecord]:
    """기록 파일(jsonl)을 읽는다. 목록 값은 tuple 로 되돌린다."""
    records: list[ResultRecord] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            spec = {
                key: tuple(value) if isinstance(value, list) else value
                for key, value in raw["spec"].items()
            }
            records.append(ResultRecord(**{**raw, "spec": spec}))
    return records


def _json_row(record: ResultRecord) -> dict[str, object]:
    """기록 한 줄 (필드 순서 그대로). 읽기 전용 매핑은 JSON 이 받는 dict 로 푼다."""
    row: dict[str, object] = {f.name: getattr(record, f.name) for f in fields(record)}
    return row | {"scores": dict(record.scores), "spec": dict(record.spec)}
