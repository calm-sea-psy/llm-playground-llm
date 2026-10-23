"""LLM 호출 예외. 쓰는 쪽은 LLMError 하나로 잡거나 원인별로 나눠 잡는다."""


class LLMError(Exception):
    """playground_llm 의 모든 예외."""


class ModelNotFoundError(LLMError):
    """서버에 그 모델이 없다 (Ollama 에 pull 하지 않은 태그 등)."""


class GenerationError(LLMError):
    """서버가 생성에 실패했다 (연결 실패 · 오류 응답 · 형식이 맞지 않는 출력)."""


class GenerationTimeoutError(GenerationError, TimeoutError):
    """정해 둔 시간 안에 생성이 끝나지 않았다."""
