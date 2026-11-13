from dataclasses import dataclass

from playground_llm.context import keep_best
from playground_llm.context.ranking import merge_best


@dataclass(frozen=True)
class Item:
    key: str | None
    score: float
    label: str = ""


def test_merge_best_keeps_first_position_and_reports_what_it_dropped():
    found = [Item("a", 0.5, "앞"), Item("b", 0.2)]
    new = [Item("a", 0.9, "새"), Item("b", 0.2, "동점"), Item("c", 0.1)]

    kept, dropped = merge_best(found, new, key=lambda item: item.key)

    assert kept == [Item("a", 0.9, "새"), Item("b", 0.2), Item("c", 0.1)]
    assert dropped == [Item("a", 0.5, "앞"), Item("b", 0.2, "동점")]


def test_merge_best_never_compares_items_without_key():
    items = [Item(None, 0.1), Item(None, 0.9)]

    assert merge_best((), items, key=lambda item: item.key) == (items, [])


def test_keep_best_is_merge_best_without_the_dropped_items():
    found = [Item("a", 0.5)]
    new = [Item("a", 0.9), Item("b", 0.1)]

    assert keep_best(found, new, key=lambda item: str(item.key)) == (Item("a", 0.9), Item("b", 0.1))
