"""결과 표를 markdown 으로 만든다."""

from collections.abc import Mapping, Sequence

DECIMALS = 3  # 0~1 지표를 0.1%p 단위까지 — 그보다 작은 차이는 문항 수가 수백 건이면 뜻이 없다


def render_table(rows: Sequence[Mapping[str, str | int | float]], columns: Sequence[str]) -> str:
    """행 목록을 markdown 표로 바꾼다. 실수는 소수점 3자리, 없는 칸은 비운다."""
    if not columns:
        raise ValueError("columns must not be empty")
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    lines.extend(
        "| " + " | ".join(_cell(row.get(column, "")) for column in columns) + " |" for row in rows
    )
    return "\n".join(lines) + "\n"


def _cell(value: str | int | float) -> str:
    if isinstance(value, float):
        return f"{value:.{DECIMALS}f}"
    return str(value)
