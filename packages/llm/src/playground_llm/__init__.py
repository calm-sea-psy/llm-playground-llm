"""LLM 호출 계약 · 타입 · 예외 · 테스트용 FakeLLM.

extras 모듈은 따로 가져온다: `from playground_llm.ollama import OllamaLLM` (`[ollama]`),
`from playground_llm.structured import generate_json` (`[json]`).
컨텍스트 엔진(칸 · 예산 · 포장 · 이력 · 출처 범위 검사)은 `from playground_llm.context import ContextAssembler`
처럼 하위 패키지에서 가져온다 (extras 없음).
"""

from playground_llm.errors import (
    GenerationError,
    GenerationTimeoutError,
    LLMError,
    ModelNotFoundError,
    StructuredOutputError,
)
from playground_llm.fake import FakeCall, FakeLLM
from playground_llm.prompts import Prompt, PromptError, PromptLibrary, PromptTemplate, compose
from playground_llm.protocols import LLM
from playground_llm.types import (
    DONE_REASONS,
    ROLES,
    THINK,
    DoneReason,
    Generation,
    Message,
    ModelSpec,
    ParamValue,
    Role,
    StreamDelta,
    Think,
    Usage,
)

__all__ = [
    "DONE_REASONS",
    "LLM",
    "ROLES",
    "THINK",
    "DoneReason",
    "FakeCall",
    "FakeLLM",
    "Generation",
    "GenerationError",
    "GenerationTimeoutError",
    "LLMError",
    "Message",
    "ModelNotFoundError",
    "ModelSpec",
    "ParamValue",
    "Prompt",
    "PromptError",
    "PromptLibrary",
    "PromptTemplate",
    "Role",
    "StreamDelta",
    "StructuredOutputError",
    "Think",
    "Usage",
    "compose",
]
