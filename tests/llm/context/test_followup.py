from playground_llm.context import Turn, clarification_questions, search_query


def test_clarification_questions_take_the_trailing_run_only():
    turns = [
        Turn("답이 나온 질문", "답", clarification=True),
        Turn("다른 질문", "답"),
        Turn("세종 때 궁궐은?", "어느 궁궐인가요?", clarification=True),
        Turn("임금이 머문 곳", "언제쯤인가요?", clarification=True),
    ]

    assert clarification_questions(turns) == ["세종 때 궁궐은?", "임금이 머문 곳"]
    assert clarification_questions([*turns, Turn("1420년", "경복궁입니다.")]) == []
    assert clarification_questions([]) == []


def test_search_query_prepends_prior_questions_and_drops_the_oldest_first():
    prior = ["가가가", "나나"]

    assert search_query("다", prior, max_chars=100) == "가가가 나나 다"
    assert search_query("다", prior, max_chars=len("나나 다")) == "나나 다"
    assert search_query("다", [], max_chars=100) == "다"
    assert search_query("아주 긴 질문", prior, max_chars=3) == "아주 긴 질문"
