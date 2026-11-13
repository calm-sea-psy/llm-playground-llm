"""컨텍스트 조립: 후보를 칸별로 골라(중복 제거 · 점수순 · 예산) 배치 순서대로 포장해 대화를 만든다.

근거는 요약하지 않고 고른다 — 요약하면 출처 번호와 원문의 연결이 끊기고, 요약하다 사실이 틀어지면 지어냄이 된다.
이전 대화(history 칸)는 근거가 아니라 맥락이라 몫을 넘는 답의 뒤를 자른다.
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

from playground_llm.context.packing import (
    TagScrubber,
    render_data_slot,
    render_history_slot,
    render_turn,
    slot_frame,
)
from playground_llm.context.ranking import merge_best
from playground_llm.context.slots import Candidate, ContextLayout, Slot, Turn
from playground_llm.context.tokens import TokenCounter
from playground_llm.errors import LLMError
from playground_llm.types import Message, Role

SLOT_SEPARATOR = "\n\n"
"""같은 턴에 이어지는 칸 사이."""
ELLIPSIS = "…"
"""history 칸에서 몫을 넘어 자른 답 끝에 붙이는 표시."""

DropReason = Literal["duplicate", "group_limit", "over_budget"]
"""후보를 뺀 이유. duplicate 는 같은 key 의 낮은 점수, group_limit 은 묶음 상한, over_budget 은 남은 예산 부족."""


class BudgetError(LLMError, ValueError):
    """반드시 넣어야 하는 칸(지시 · 필수 data 칸)이 남은 예산에 들어가지 않는다."""


@dataclass(frozen=True)
class Placed:
    """칸에 넣은 후보 하나."""

    candidate: Candidate
    number: int | None
    """번호 칸이면 1부터의 번호 (답 속 `[n]`), 아니면 None."""


@dataclass(frozen=True)
class Dropped:
    """선별에서 뺀 후보 하나. 로그 · 평가에 쓴다."""

    candidate: Candidate
    reason: DropReason


@dataclass(frozen=True)
class SlotResult:
    """포장을 마친 칸 하나."""

    slot: Slot
    text: str
    """태그로 감싼 글 (지시 칸은 지시 그대로). 비었으면 빈 글 — 대화에 넣지 않는다."""
    tokens: int
    """text 를 조립기의 TokenCounter 로 센 값."""
    placed: tuple[Placed, ...]
    """넣은 후보, 놓인 순서대로. 지시 · history 칸은 비어 있다."""
    turns: tuple[Turn, ...] = ()
    """history 칸에 넣은 턴, 대화 순서대로. 자른 답은 자른 그대로 — 모델이 본 글과 같다."""


@dataclass(frozen=True)
class AssembledContext:
    """LLM 호출 한 번에 넘길 컨텍스트."""

    messages: tuple[Message, ...]
    """배치 순서대로 칸을 이은 대화. 같은 턴이 이어지는 칸은 한 메시지로 합친다."""
    slots: tuple[SlotResult, ...]
    """배치 순서대로. 빈 칸도 들어 있다."""
    dropped: tuple[Dropped, ...]
    """뺀 후보 — 칸 우선순위 순서, 칸 안에서는 고른 순서."""
    tokens: int
    """칸 토큰의 합. 턴 표지 · 칸 사이 빈 줄은 세지 않는다 (예산에 여유로 둔다)."""
    budget: int
    """레이아웃의 전체 상한."""
    dropped_turns: tuple[Turn, ...] = ()
    """history 칸에 넣지 못한 턴 (받은 그대로), 대화 순서대로."""

    def slot(self, name: str) -> SlotResult:
        """이름이 name 인 칸의 결과를 돌려준다. 프롬프트 틀에 칸 글을 따로 넣을 때 쓴다.

        Raises:
          KeyError: 그런 칸이 없을 때.
        """
        for result in self.slots:
            if result.slot.name == name:
                return result
        raise KeyError(name)


class ContextAssembler:
    """칸 배치대로 후보를 골라 예산에 맞추고, 태그로 포장해 대화를 만든다.

    칸은 우선순위 순서로 예산을 받는다: 칸마다 min(칸 예산, 남은 전체 예산) 안에서 고른다.
    data 칸은 중복 제거 → 묶음 상한 → 점수 높은 것부터 들어가는 만큼 (안 들어가는 항목은 건너뛰고 다음 항목을 본다).
    history 칸은 받은 턴이 예산을 똑같이 나눈다 — 몫을 넘는 턴은 답 뒤를 자르고, 질문만으로 넘으면 뺀다.
    이력이 근거를 밀어내지 않게 history 칸은 문서 칸보다 우선순위를 낮게(숫자를 크게) 둔다.

    예)
        assembler = ContextAssembler(layout, CharTokenCounter())
        context = assembler.assemble({"system": system_text}, candidates, history=turns)
        llm.generate(spec, context.messages)
    """

    def __init__(self, layout: ContextLayout, counter: TokenCounter) -> None:
        self._layout = layout
        self._counter = counter
        self._scrubber = TagScrubber(layout.tags)

    def assemble(
        self,
        instructions: Mapping[str, str],
        candidates: Iterable[Candidate],
        *,
        history: Sequence[Turn] = (),
    ) -> AssembledContext:
        """지시 · 후보 · 이전 턴으로 컨텍스트를 만든다.

        Args:
          instructions: 지시 칸 이름 → 지시 글 (프롬프트 파일을 채운 것). 빠진 지시 칸은 빈 칸.
          candidates: data 칸에 넣을 후보. 같은 칸 안에서는 받은 순서가 `order="given"` 의 순서다.
          history: history 칸에 넣을 앞선 턴, 대화 순서대로 (오래된 것 → 최신). 넣을 최근 턴만 넘긴다
            (예: 저장소에서 최근 3턴). 레이아웃에 history 칸이 없으면 비워 둔다.

        Raises:
          BudgetError: 지시 칸 · 필수 data 칸이 남은 예산에 들어가지 않을 때.
        """
        by_slot = self._group(instructions, candidates)
        if history and not any(slot.kind == "history" for slot in self._layout.slots):
            raise ValueError("history was given but this layout has no history slot")
        remaining = self._layout.budget
        results: dict[str, SlotResult] = {}
        dropped: list[Dropped] = []
        dropped_turns: list[Turn] = []
        # sorted 는 안정 정렬 — 우선순위가 같으면 배치 순서대로 예산을 받는다
        for slot in sorted(self._layout.slots, key=lambda slot: slot.priority):
            available = min(slot.budget, remaining)
            if slot.kind == "instruction":
                result = self._instruction(slot, instructions.get(slot.name, ""), available)
            elif slot.kind == "history":
                result, slot_dropped_turns = self._history(slot, history, available)
                dropped_turns.extend(slot_dropped_turns)
            else:
                result, slot_dropped = self._select(slot, by_slot.get(slot.name, []), available)
                dropped.extend(slot_dropped)
            remaining -= result.tokens
            results[slot.name] = result
        ordered = tuple(results[slot.name] for slot in self._layout.slots)
        return AssembledContext(
            messages=_messages(ordered),
            slots=ordered,
            dropped=tuple(dropped),
            tokens=sum(result.tokens for result in ordered),
            budget=self._layout.budget,
            dropped_turns=tuple(dropped_turns),
        )

    def _group(
        self, instructions: Mapping[str, str], candidates: Iterable[Candidate]
    ) -> dict[str, list[Candidate]]:
        kinds = {slot.name: slot.kind for slot in self._layout.slots}
        for name in instructions:
            if kinds.get(name) != "instruction":
                raise ValueError(f"{name!r} is not an instruction slot of this layout")
        by_slot: dict[str, list[Candidate]] = {}
        for candidate in candidates:
            # 바깥에서 온 글이 지시 칸에 들어가는 길을 막는다 — 후보는 태그 안(data 칸)으로만.
            # history 칸은 `Turn` 으로만 채운다 (후보로 넣으면 턴 몫 나누기 · 답 자르기를 건너뛴다)
            if kinds.get(candidate.slot) != "data":
                raise ValueError(f"{candidate.slot!r} is not a data slot of this layout")
            by_slot.setdefault(candidate.slot, []).append(candidate)
        return by_slot

    def _instruction(self, slot: Slot, text: str, available: int) -> SlotResult:
        tokens = self._counter.count(text)
        if tokens > available:
            raise BudgetError(f"{slot.name}: instruction needs {tokens} tokens, {available=}")
        return SlotResult(slot=slot, text=text, tokens=tokens, placed=())

    def _select(
        self, slot: Slot, candidates: Sequence[Candidate], available: int
    ) -> tuple[SlotResult, list[Dropped]]:
        unique, dropped = _dedupe(candidates)
        # (받은 순서, 후보) — given 배치는 받은 순서로 되돌린다. 점수 내림차순, 같으면 받은 순서
        ranked = sorted(enumerate(unique), key=lambda pair: -pair[1].score)
        chosen: list[tuple[int, Candidate]] = []
        groups: Counter[str] = Counter()
        text, tokens = "", 0
        for position, candidate in ranked:
            group = candidate.group
            limit = slot.max_per_group
            if limit is not None and group is not None and groups[group] >= limit:
                dropped.append(Dropped(candidate, "group_limit"))
                continue
            trial = [*chosen, (position, candidate)]
            trial_text = render_data_slot(slot, _arrange(slot, trial), self._scrubber)
            trial_tokens = self._counter.count(trial_text)
            if trial_tokens > available:
                if slot.required:
                    raise BudgetError(
                        f"{slot.name}: required item does not fit ({trial_tokens=}, {available=})"
                    )
                dropped.append(Dropped(candidate, "over_budget"))
                continue
            chosen, text, tokens = trial, trial_text, trial_tokens
            if group is not None:
                groups[group] += 1
        placed = tuple(Placed(candidate, number) for number, candidate in _arrange(slot, chosen))
        return SlotResult(slot=slot, text=text, tokens=tokens, placed=placed), dropped

    def _history(
        self, slot: Slot, turns: Sequence[Turn], available: int
    ) -> tuple[SlotResult, list[Turn]]:
        """턴마다 같은 몫 — 긴 답 하나가 다른 턴을 밀어내지 않게. 칸 태그 · 빈 줄 몫은 먼저 뺀다."""
        if not turns:
            return SlotResult(slot=slot, text="", tokens=0, placed=()), []
        frame = self._counter.count(slot_frame(slot, len(turns)))
        share = (available - frame) // len(turns)
        kept: list[tuple[int, Turn]] = []  # (받은 자리, 넣을 턴)
        for position, turn in enumerate(turns):
            fitted = self._fit_turn(slot, turn, share)
            if fitted is not None:
                kept.append((position, fitted))
        text = render_history_slot(slot, [turn for _, turn in kept], self._scrubber)
        tokens = self._counter.count(text)
        # 조각마다 몫 안이어도 이어 붙인 글을 더 크게 세는 토크나이저가 있다 — 넘으면 오래된 턴부터 뺀다
        while kept and tokens > available:
            kept.pop(0)
            text = render_history_slot(slot, [turn for _, turn in kept], self._scrubber)
            tokens = self._counter.count(text)
        positions = {position for position, _ in kept}
        dropped = [turn for position, turn in enumerate(turns) if position not in positions]
        placed = tuple(turn for _, turn in kept)
        return SlotResult(slot=slot, text=text, tokens=tokens, placed=(), turns=placed), dropped

    def _fit_turn(self, slot: Slot, turn: Turn, share: int) -> Turn | None:
        """몫 안에 드는 턴을 돌려준다 — 넘으면 답의 가장 긴 앞부분 + `…`, 질문만으로 넘으면 None."""
        if self._turn_tokens(slot, turn) <= share:
            return turn
        if self._turn_tokens(slot, _cut(turn, 0)) > share:
            return None
        # 몫에 드는 가장 긴 앞부분 길이 (이분 탐색). 길이 0 은 위에서 들어가는 것을 확인했다
        low, high = 0, len(turn.answer) - 1
        while low < high:
            middle = (low + high + 1) // 2
            if self._turn_tokens(slot, _cut(turn, middle)) <= share:
                low = middle
            else:
                high = middle - 1
        return _cut(turn, low)

    def _turn_tokens(self, slot: Slot, turn: Turn) -> int:
        return self._counter.count(render_turn(slot, turn, self._scrubber))


def _dedupe(candidates: Sequence[Candidate]) -> tuple[list[Candidate], list[Dropped]]:
    """같은 key 는 점수가 가장 높은 하나만 (`merge_best` 규칙). key 가 없는 후보는 비교하지 않는다."""
    unique, duplicates = merge_best((), candidates, key=lambda candidate: candidate.key)
    return unique, [Dropped(candidate, "duplicate") for candidate in duplicates]


def _arrange(
    slot: Slot, chosen: Sequence[tuple[int, Candidate]]
) -> list[tuple[int | None, Candidate]]:
    """고른 후보를 놓을 순서로 늘어놓고 번호를 붙인다 (chosen 은 점수 순서)."""
    ordered = sorted(chosen, key=lambda pair: pair[0]) if slot.order == "given" else list(chosen)
    if not slot.numbered:
        return [(None, candidate) for _, candidate in ordered]
    return [(number, candidate) for number, (_, candidate) in enumerate(ordered, start=1)]


def _cut(turn: Turn, length: int) -> Turn:
    """답을 앞 length 글자만 남기고 `…` 를 붙인 턴."""
    return replace(turn, answer=turn.answer[:length].rstrip() + ELLIPSIS)


def _messages(results: Sequence[SlotResult]) -> tuple[Message, ...]:
    """빈 칸은 빼고, 같은 턴이 이어지는 칸은 한 메시지로."""
    turns: list[tuple[Role, list[str]]] = []
    for result in results:
        if not result.text:
            continue
        if turns and turns[-1][0] == result.slot.role:
            turns[-1][1].append(result.text)
        else:
            turns.append((result.slot.role, [result.text]))
    return tuple(Message(role, SLOT_SEPARATOR.join(texts)) for role, texts in turns)
