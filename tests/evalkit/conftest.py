import json
from pathlib import Path

import pytest


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
