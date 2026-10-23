"""LLM 호출 계약 · 타입 · 예외 · 테스트용 FakeLLM.

서버 클라이언트는 extras 모듈에서 가져온다: `from playground_llm.ollama import OllamaLLM` (`[ollama]`).
"""

from playground_llm.errors import (
    GenerationError,
    GenerationTimeoutError,
    LLMError,
    ModelNotFoundError,
)
from playground_llm.fake import FakeCall, FakeLLM
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
    "Role",
    "StreamDelta",
    "Usage",
]
