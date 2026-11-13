"""컨텍스트 엔진: 칸 정의 · 예산 · 선별 · 배치 · 포장 · 멀티턴 이력 · 단기 메모리 · 후보 출처 약속 · 출처 범위 검사.

LLM 을 한 번 부를 때 넘길 입력(지시 · 검색 문서 · 도구 결과 · 기억 · 이전 대화 · 질문)을 칸별 예산 안에서 골라
대화로 만들고, 모델이 고른 출처 번호가 그 호출의 문맥 안에 있는지 확인한다.
표준 라이브러리와 playground_llm 의 타입 · 예외만 쓴다 — 어떤 칸을 둘지(칸 이름 · 예산 · 태그)는 쓰는 제품 레포가 정한다.
"""

from playground_llm.context.assembler import (
    ELLIPSIS,
    SLOT_SEPARATOR,
    AssembledContext,
    BudgetError,
    ContextAssembler,
    Dropped,
    DropReason,
    Placed,
    SlotResult,
)
from playground_llm.context.citations import UnknownSourceError, cited_sources, unknown_sources
from playground_llm.context.followup import (
    QUERY_SEPARATOR,
    clarification_questions,
    search_query,
)
from playground_llm.context.memory import Evidence, WorkingMemory, memory_candidates
from playground_llm.context.packing import ITEM_SEPARATOR, TagScrubber
from playground_llm.context.ranking import keep_best
from playground_llm.context.slots import (
    ASSISTANT_TAG,
    ITEM_ORDERS,
    NUMBER_ATTRIBUTE,
    SLOT_KINDS,
    USER_TAG,
    Candidate,
    ContextLayout,
    ItemOrder,
    Slot,
    SlotKind,
    Turn,
)
from playground_llm.context.sources import CandidateSource, collect_candidates
from playground_llm.context.tokens import (
    TOKENS_PER_CHAR,
    CharTokenCounter,
    TokenCounter,
)

__all__ = [
    "ASSISTANT_TAG",
    "ELLIPSIS",
    "ITEM_ORDERS",
    "ITEM_SEPARATOR",
    "NUMBER_ATTRIBUTE",
    "QUERY_SEPARATOR",
    "SLOT_KINDS",
    "SLOT_SEPARATOR",
    "TOKENS_PER_CHAR",
    "USER_TAG",
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
    "Turn",
    "UnknownSourceError",
    "WorkingMemory",
    "cited_sources",
    "clarification_questions",
    "collect_candidates",
    "keep_best",
    "memory_candidates",
    "search_query",
    "unknown_sources",
]
