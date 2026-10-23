"""LLM 호출 계약 · 타입 · 예외 · 테스트용 FakeLLM.

extras 모듈은 따로 가져온다: `from playground_llm.ollama import OllamaLLM` (`[ollama]`),
`from playground_llm.structured import generate_json` (`[json]`).
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
    ROLES,
    THINK,
    Generation,
    Message,
    ModelSpec,
    ParamValue,
    Role,
    StreamDelta,
    Usage,
)

__all__ = [
    "LLM",
    "ROLES",
    "THINK",
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
    "Usage",
    "compose",
]
