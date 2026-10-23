from pathlib import Path

import pytest

from playground_llm import PromptError, PromptLibrary, PromptTemplate, compose

BASE = PromptTemplate("base", "v1", "너는 위키 QA 도우미다. 출처는 [문서 제목] 으로 적는다.")
GROUNDED = PromptTemplate(
    "grounded", "v2", '문맥:\n$context\n\n질문: ${question}\n출력 예: {"answer": "…"}'
)


def test_template_lists_variables_and_renders_keeping_json_braces():
    assert GROUNDED.variables == {"context", "question"}
    assert GROUNDED.ref == "grounded@v2"
    assert (
        GROUNDED.render(context="C", question="Q")
        == '문맥:\nC\n\n질문: Q\n출력 예: {"answer": "…"}'
    )


def test_render_rejects_missing_and_unused_values():
    with pytest.raises(PromptError, match=r"missing values \['question'\]"):
        GROUNDED.render(context="C")
    with pytest.raises(PromptError, match=r"unused values \['qestion'\]"):
        GROUNDED.render(context="C", question="Q", qestion="오타")


def test_template_rejects_bad_names_versions_and_placeholders():
    with pytest.raises(PromptError, match="snake_case"):
        PromptTemplate("Grounded", "v1", "x")
    with pytest.raises(PromptError, match="v1"):
        PromptTemplate("grounded", "1", "x")
    with pytest.raises(PromptError, match=r"\$\$"):
        PromptTemplate("grounded", "v1", "가격은 $5")


def test_dollar_sign_is_written_as_double_dollar():
    assert PromptTemplate("p", "v1", "가격은 $$5").render() == "가격은 $5"


def test_compose_joins_parts_in_order_with_versions_and_digests():
    prompt = compose([BASE, GROUNDED], {"context": "C", "question": "Q"})

    assert prompt.text.startswith(BASE.text + "\n\n문맥:\nC")
    assert prompt.version == f"base@v1:{BASE.digest}+grounded@v2:{GROUNDED.digest}"


def test_digest_changes_when_text_changes_without_version_bump():
    edited = PromptTemplate("base", "v1", BASE.text + " 짧게 답한다.")

    assert edited.ref == BASE.ref
    assert edited.digest != BASE.digest
    assert len(BASE.digest) == 8


def test_compose_checks_values_across_all_parts():
    with pytest.raises(PromptError, match="missing values"):
        compose([BASE, GROUNDED], {"context": "C"})
    with pytest.raises(PromptError, match="unused values"):
        compose([BASE], {"context": "C"})
    with pytest.raises(PromptError, match="must not be empty"):
        compose([], {})


def test_library_loads_name_version_files(tmp_path: Path):
    for name, version, text in [
        ("base", "v1", BASE.text),
        ("grounded", "v1", "옛 버전 $question"),
        ("grounded", "v2", GROUNDED.text),
    ]:
        (tmp_path / name).mkdir(exist_ok=True)
        (tmp_path / name / f"{version}.md").write_text(text + "\n", encoding="utf-8")
    (tmp_path / "README.txt").write_text("무시한다", encoding="utf-8")

    library = PromptLibrary.load(tmp_path)

    assert library.refs == ("base@v1", "grounded@v1", "grounded@v2")
    assert library.get("grounded@v2") == GROUNDED  # 끝 줄바꿈은 뗀다
    with pytest.raises(PromptError, match="unknown prompt 'grounded'"):
        library.get("grounded")  # 버전 생략 불가


def test_library_rejects_missing_directory_and_bad_file_names(tmp_path: Path):
    with pytest.raises(PromptError, match="not found"):
        PromptLibrary.load(tmp_path / "없음")
    (tmp_path / "base").mkdir()
    (tmp_path / "base" / "latest.md").write_text("x", encoding="utf-8")
    with pytest.raises(PromptError, match="v1"):
        PromptLibrary.load(tmp_path)


def test_library_rejects_duplicates():
    with pytest.raises(PromptError, match="duplicate"):
        PromptLibrary([BASE, BASE])
