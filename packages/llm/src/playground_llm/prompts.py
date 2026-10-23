"""프롬프트 템플릿: 파일에서 읽기 · 값 채우기 · 버전. 프롬프트 텍스트는 쓰는 제품 레포가 갖고, 여기는 다루는 법만.

폴더 구조는 `{이름}/{버전}.md` (예: `prompts/grounded/v2.md`). 자리표시자는 `$name` 또는 `${name}` —
JSON 예시의 중괄호를 그대로 쓸 수 있게 `str.format` 대신 `string.Template` 을 쓴다. `$` 자체는 `$$` 로 쓴다.
"""

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Self

from playground_llm.errors import LLMError

PROMPT_SUFFIX = ".md"
DIGEST_CHARS = 8  # 내용 해시 앞 8자 — 버전을 안 올리고 파일만 고친 것을 기록에서 구분하는 용도
SEPARATOR = "\n\n"
_VERSION = re.compile(r"^v\d+$")
_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class PromptError(LLMError):
    """프롬프트를 찾을 수 없거나 값을 채울 수 없다."""


@dataclass(frozen=True)
class PromptTemplate:
    """프롬프트 파일 하나 (이름 · 버전 · 본문).

    예)
        template = PromptTemplate("grounded", "v1", "문맥: $context / 질문: $question")
        template.ref        # "grounded@v1"
        template.render(context="…", question="…")
    """

    name: str
    version: str
    """`v` + 숫자. 내용을 바꾸면 올린다."""
    text: str

    def __post_init__(self) -> None:
        if not _NAME.match(self.name):
            raise PromptError(f"name must be snake_case, got {self.name=}")
        if not _VERSION.match(self.version):
            raise PromptError(f"version must look like v1, got {self.version=}")
        if not Template(self.text).is_valid():  # `$` 뒤 숫자 · 공백 등 — 만들 때 바로 알린다
            raise PromptError(f"{self.ref}: invalid placeholder (write a literal $ as $$)")

    @property
    def ref(self) -> str:
        """`이름@버전`. 프로필 표가 프롬프트를 가리키는 이름."""
        return f"{self.name}@{self.version}"

    @property
    def digest(self) -> str:
        """본문 sha256 앞 8자."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:DIGEST_CHARS]

    @property
    def variables(self) -> frozenset[str]:
        """채워야 하는 자리표시자 이름."""
        return frozenset(Template(self.text).get_identifiers())

    def render(self, **values: str) -> str:
        """자리표시자를 채운 본문.

        Raises:
          PromptError: 채우지 않은 자리표시자가 있거나, 쓰지 않는 값을 넘겼을 때.
        """
        _check_values(self.ref, self.variables, values)
        return Template(self.text).substitute(values)


@dataclass(frozen=True)
class Prompt:
    """값을 채운 프롬프트와 그 버전."""

    text: str
    version: str
    """조각마다 `이름@버전:해시` 를 `+` 로 이은 것 (예: `base@v1:1a2b3c4d+grounded@v2:5e6f7a8b`).
    평가 기록의 prompt_version 으로 그대로 쓴다."""


def compose(
    parts: Sequence[PromptTemplate], values: Mapping[str, str], *, separator: str = SEPARATOR
) -> Prompt:
    """조각들(공통 베이스 → 단계 블록 …)을 순서대로 채워 이어 붙인다.

    공통 부분을 앞에 두면 서버의 prefix 캐시가 그만큼을 요청 사이에 다시 계산하지 않는다.

    Raises:
      PromptError: 어떤 조각의 자리표시자가 values 에 없거나, 어느 조각도 쓰지 않는 값이 있을 때.
    """
    if not parts:
        raise PromptError("parts must not be empty")
    needed = frozenset().union(*(part.variables for part in parts))
    _check_values("+".join(part.ref for part in parts), needed, values)
    text = separator.join(Template(part.text).substitute(values) for part in parts)
    version = "+".join(f"{part.ref}:{part.digest}" for part in parts)
    return Prompt(text=text, version=version)


class PromptLibrary:
    """폴더에서 읽은 프롬프트 모음. `이름@버전` 으로 꺼낸다.

    예)
        library = PromptLibrary.load(Path("prompts"))
        prompt = compose([library.get("base@v1"), library.get("grounded@v2")], {"context": …})
    """

    def __init__(self, templates: Sequence[PromptTemplate]) -> None:
        by_ref: dict[str, PromptTemplate] = {}
        for template in templates:
            if template.ref in by_ref:
                raise PromptError(f"duplicate prompt {template.ref}")
            by_ref[template.ref] = template
        self._by_ref = by_ref

    @classmethod
    def load(cls, directory: Path) -> Self:
        """`{이름}/{버전}.md` 파일을 모두 읽는다. 끝 줄바꿈은 뗀다.

        Raises:
          PromptError: 폴더가 없거나 이름 · 버전 형식이 틀린 파일이 있을 때.
        """
        if not directory.is_dir():
            raise PromptError(f"prompt directory not found: {directory}")
        templates = [
            PromptTemplate(
                name=path.parent.name,
                version=path.stem,
                text=path.read_text(encoding="utf-8").rstrip("\n"),
            )
            for path in sorted(directory.glob(f"*/*{PROMPT_SUFFIX}"))
        ]
        return cls(templates)

    @property
    def refs(self) -> tuple[str, ...]:
        """가진 프롬프트의 `이름@버전` (정렬)."""
        return tuple(sorted(self._by_ref))

    def get(self, ref: str) -> PromptTemplate:
        """`이름@버전` 의 템플릿. 버전은 생략할 수 없다 — 벤치 · 기록이 가리키는 버전이 바뀌지 않게.

        Raises:
          PromptError: 없는 ref 일 때.
        """
        if ref not in self._by_ref:
            raise PromptError(f"unknown prompt {ref!r}, known: {', '.join(self.refs)}")
        return self._by_ref[ref]


def _check_values(ref: str, needed: frozenset[str], values: Mapping[str, str]) -> None:
    if missing := needed - values.keys():
        raise PromptError(f"{ref}: missing values {sorted(missing)}")
    if unused := values.keys() - needed:  # 이름을 잘못 쓴 값은 조용히 버려지기 쉽다
        raise PromptError(f"{ref}: unused values {sorted(unused)}")
