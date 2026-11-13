import pytest

from playground_llm import LLMError
from playground_llm.context import (
    AssembledContext,
    Candidate,
    CharTokenCounter,
    ContextAssembler,
    ContextLayout,
    Slot,
    UnknownSourceError,
    cited_sources,
    unknown_sources,
)

LAYOUT = ContextLayout(
    (
        Slot("docs", "data", budget=1000, item_tag="document", numbered=True, order="given"),
        Slot("question", "data", budget=100, tag="question"),
    ),
    budget=2000,
)


def _context(*titles: str) -> AssembledContext:
    candidates = [
        Candidate("docs", f"{title} 본문", attributes={"title": title}) for title in titles
    ]
    return ContextAssembler(LAYOUT, CharTokenCounter()).assemble({}, candidates)


def test_cited_sources_come_from_this_context_in_model_order_without_repeats():
    docs = _context("경복궁", "창덕궁", "덕수궁").slot("docs")

    cited = cited_sources([3, 1, 3], docs)

    assert [placed.number for placed in cited] == [3, 1]
    assert [placed.candidate.attributes["title"] for placed in cited] == ["덕수궁", "경복궁"]
    assert cited_sources([], docs) == ()


def test_numbers_from_an_earlier_context_are_rejected():
    _context("경복궁", "창덕궁", "덕수궁")  # 앞 호출 — 번호 3 까지 있었다
    docs = _context("경복궁", "창덕궁").slot("docs")

    with pytest.raises(UnknownSourceError) as caught:
        cited_sources([1, 3, 99], docs)

    assert caught.value.unknown == (3, 99)
    assert str(caught.value) == "source numbers not in this context: [3, 99]"
    assert unknown_sources([1, 3, 99], docs) == [3, 99]
    assert unknown_sources([2, 1], docs) == []


def test_unknown_source_error_is_an_llm_error_and_a_value_error():
    assert issubclass(UnknownSourceError, LLMError)
    assert issubclass(UnknownSourceError, ValueError)


def test_sources_must_come_from_a_numbered_slot():
    context = _context("경복궁")

    with pytest.raises(ValueError, match="numbered slot"):
        cited_sources([1], context.slot("question"))
