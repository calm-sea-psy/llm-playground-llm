"""LLM 호출 예외. 쓰는 쪽은 LLMError 하나로 잡거나 원인별로 나눠 잡는다."""

from playground_llm.types import DoneReason


class LLMError(Exception):
    """playground_llm 의 모든 예외."""


class ModelNotFoundError(LLMError):
    """서버에 그 모델이 없다 (Ollama 에 pull 하지 않은 태그 등)."""


class GenerationError(LLMError):
    """서버가 생성에 실패했다 (연결 실패 · 오류 응답 · 형식이 맞지 않는 출력)."""


class GenerationTimeoutError(GenerationError, TimeoutError):
    """정해 둔 시간 안에 생성이 끝나지 않았다."""


class StructuredOutputError(GenerationError):
    """다시 시도해도 출력이 요구한 JSON 형식에 맞지 않았다 (또는 max_tokens 에 닿아 잘렸다)."""

    def __init__(
        self,
        message: str,
        *,
        last_text: str,
        attempts: int,
        done_reason: DoneReason | None = None,
    ) -> None:
        super().__init__(message)
        self.last_text = last_text
        """마지막 출력 원문 (로그 · 디버그용)."""
        self.attempts = attempts
        self.done_reason = done_reason
        """마지막 생성이 멈춘 이유. length 면 형식이 아니라 토큰 한도 문제다 (thinking 이 다 쓴 경우 포함)."""
