"""컨텍스트 엔진: 칸 정의 · 예산 · 선별 · 배치 · 포장 · 단기 메모리 · 후보 출처 약속.

LLM 을 한 번 부를 때 넘길 입력(지시 · 검색 문서 · 도구 결과 · 기억 · 질문)을 칸별 예산 안에서 골라 대화로 만든다.
표준 라이브러리와 playground_llm 의 타입 · 예외만 쓴다 — 어떤 칸을 둘지(칸 이름 · 예산 · 태그)는 쓰는 제품 레포가 정한다.
"""

from playground_llm.context.assembler import (
    SLOT_SEPARATOR,
    AssembledContext,
    BudgetError,
    ContextAssembler,
    Dropped,
    DropReason,
    Placed,
    SlotResult,
)
from playground_llm.context.memory import (
    Evidence,
    WorkingMemory,
    keep_best,
    memory_candidates,
)
from playground_llm.context.packing import ITEM_SEPARATOR, TagScrubber
from playground_llm.context.slots import (
    ITEM_ORDERS,
    NUMBER_ATTRIBUTE,
    SLOT_KINDS,
    Candidate,
    ContextLayout,
    ItemOrder,
    Slot,
    SlotKind,
)
from playground_llm.context.sources import CandidateSource, collect_candidates
from playground_llm.context.tokens import (
    TOKENS_PER_CHAR,
    CharTokenCounter,
    TokenCounter,
)

__all__ = [
    "ITEM_ORDERS",
    "ITEM_SEPARATOR",
    "NUMBER_ATTRIBUTE",
    "SLOT_KINDS",
    "SLOT_SEPARATOR",
    "TOKENS_PER_CHAR",
    "AssembledContext",
    "BudgetError",
    "Candidate",
    "CandidateSource",
    "CharTokenCounter",
    "ContextAssembler",
    "ContextLayout",
    "DropReason",
    "Dropped",
    "Evidence",
    "ItemOrder",
    "Placed",
    "Slot",
    "SlotKind",
    "SlotResult",
    "TagScrubber",
    "TokenCounter",
    "WorkingMemory",
    "collect_candidates",
    "keep_best",
    "memory_candidates",
]
