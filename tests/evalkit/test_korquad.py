import json
from pathlib import Path

import pytest

from playground_evalkit import KorQuADItem, contains_answer, korquad_title, load_korquad


@pytest.fixture
def korquad_file(tmp_path: Path) -> Path:
    data = {
        "version": "test",
        "data": [
            {
                "title": "증강현실",
                "paragraphs": [
                    {
                        "context": "…",
                        "qas": [
                            {
                                "id": "q1",
                                "question": "기술 이름은?",
                                "answers": [{"text": "증강현실", "answer_start": 0}],
                            }
                        ],
                    }
                ],
            },
            {
                "title": "커맨드_앤_컨커_3__타이베리움_워",
                "paragraphs": [
                    {
                        "context": "…",
                        "qas": [
                            {
                                "id": "q2",
                                "question": "개발사는?",
                                "answers": [{"text": "EA", "answer_start": 0}],
                            },
                            {
                                "id": "q3",
                                "question": "출시 연도는?",
                                "answers": [{"text": "2007년", "answer_start": 0}],
                            },
                        ],
                    }
                ],
            },
        ],
    }
    path = tmp_path / "korquad.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_load_korquad_normalizes_titles_and_reads_questions_in_order(korquad_file: Path):
    items = load_korquad(korquad_file)

    assert [item.id for item in items] == ["q1", "q2", "q3"]
    assert items[1] == KorQuADItem(
        id="q2",
        title="커맨드 앤 컨커 3: 타이베리움 워",
        question="개발사는?",
        answer="EA",
        source_title="커맨드_앤_컨커_3__타이베리움_워",
    )


def test_load_korquad_applies_aliases_given_in_either_spelling(korquad_file: Path):
    items = load_korquad(korquad_file, title_aliases={"증강현실": "증강_현실"})

    assert items[0].title == "증강 현실"


def test_korquad_title_turns_double_underscore_into_colon():
    assert korquad_title("커맨드_앤_컨커_3__타이베리움_워") == "커맨드 앤 컨커 3: 타이베리움 워"


def test_contains_answer_ignores_spacing_and_needs_same_title(korquad_file: Path):
    item = load_korquad(korquad_file, title_aliases={"증강현실": "증강 현실"})[0]

    assert contains_answer("증강_현실", "증강 현실은 실제 환경에 가상 정보를 겹친다.", item)
    assert not contains_answer("가상 현실", "증강 현실은 실제 환경에 가상 정보를 겹친다.", item)
    assert not contains_answer("증강 현실", "다른 내용이다.", item)
