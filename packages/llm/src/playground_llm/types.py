"""LLM 호출의 입력 · 출력 자료형. 서버 종류(Ollama · vLLM)와 상관없이 같은 모양이다."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Self, get_args

# playground_evalkit.ParamValue 와 같은 정의 — ResultRecord.spec 이 ModelSpec.params 를 그대로 받는다.
# 두 모듈은 서로 의존하지 않으므로 한 줄을 각자 둔다
type ParamValue = str | int | float | bool | tuple[str, ...]
"""설정 값. tuple 은 Ollama `stop` 같은 문자열 목록."""

Role = Literal["system", "user", "assistant"]
ROLES: tuple[str, ...] = get_args(Role)

DoneReason = Literal["stop", "length"]
"""생성이 멈춘 이유. stop 은 모델이 스스로 끝냈거나 stop 문자열을 만난 것, length 는 max_tokens 에 닿아 잘린 것."""
DONE_REASONS: tuple[str, ...] = get_args(DoneReason)

THINK = "think"
"""thinking 설정 params 키. 값은 Think ("low" 같은 문자열도 받아 Think 로 바꾼다).
값이 다르면 spec_id 가 달라 벤치에서 따로 잰다."""


class Think(StrEnum):
    """thinking 설정. 서버로 보낼 값으로 바꾸는 건 각 클라이언트가 한다.

    LOW · MEDIUM · HIGH 는 켜되 정도를 고른 것 (gpt-oss 처럼 정도만 받는 모델용).
    모델마다 받는 값이 달라서 클라이언트가 보내기 전에 거부할 수 있다.
    """

    OFF = "off"
    ON = "on"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @property
    def enabled(self) -> bool:
        """켜지는 값인지 (정도도 켜짐). StrEnum 은 OFF 도 참이라 `if spec.think` 로 판단하면 틀린다."""
        return self is not Think.OFF


@dataclass(frozen=True)
class ModelSpec:
    """모델 하나 · 생성 설정 하나. 벤치 · 평가 기록의 키 (`spec_id`).

    예)
        spec = ModelSpec("ollama", "gemma4:12b", {"think": Think.OFF, "temperature": 0.1})
        spec.spec_id  # "ollama/gemma4:12b?temperature=0.1,think=off"
    """

    provider: str
    """서버 종류 (ollama · vllm …). 어느 클라이언트가 받을지는 쓰는 쪽이 정한다."""
    name: str
    """서버에서 쓰는 모델 이름 (태그 포함, 예: `gemma4:12b`)."""
    # 해시에서는 뺀다 (매핑은 해시가 안 된다). 같은지 비교에는 들어간다 — 같은 spec 은 해시도 같다
    params: Mapping[str, ParamValue] = field(default_factory=dict, hash=False)
    """temperature · max_tokens · num_ctx · think 등. 클라이언트가 서버 인자로 옮긴다."""

    def __post_init__(self) -> None:
        if not self.provider or not self.name:
            raise ValueError(
                f"provider and name must not be empty, got {self.provider=} {self.name=}"
            )
        params = dict(self.params)
        if THINK in params:
            params[THINK] = _think(params[THINK])
        # 만든 쪽이 넘긴 dict 를 나중에 고쳐도 spec_id 가 바뀌지 않게 읽기 전용 복사본으로
        object.__setattr__(self, "params", MappingProxyType(params))

    @property
    def spec_id(self) -> str:
        """`provider/name?키=값,…` (키 알파벳순). 설정이 같으면 같은 문자열."""
        base = f"{self.provider}/{self.name}"
        if not self.params:
            return base
        pairs = ",".join(f"{key}={_format(value)}" for key, value in sorted(self.params.items()))
        return f"{base}?{pairs}"

    @property
    def think(self) -> Think:
        """이 설정의 thinking. params 에 없으면 OFF. 켜졌는지는 `spec.think.enabled` 로 본다."""
        return Think(self.params.get(THINK, Think.OFF))

    def with_params(self, **params: ParamValue) -> Self:
        """설정값 일부를 바꾼 새 설정 (예: `spec.with_params(think=Think.ON)`)."""
        return type(self)(self.provider, self.name, {**self.params, **params})


@dataclass(frozen=True)
class Message:
    """대화 한 턴. 이전 턴의 thinking 은 넣지 않는다 — 최종 답(`Generation.text`)만."""

    role: Role
    content: str

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role=}")


@dataclass(frozen=True)
class Usage:
    """토큰 수와 시간. 서버가 주지 않는 값은 None."""

    prompt_tokens: int
    completion_tokens: int
    """생성 토큰 수. thinking 토큰도 들어 있다 (Ollama eval_count)."""
    total_ms: int
    """클라이언트가 잰 전체 시간 — 항상 있다."""
    thinking_tokens: int | None = None
    """completion_tokens 중 thinking 몫. 서버가 따로 줄 때만 (Ollama 는 주지 않는다)."""
    cached_prompt_tokens: int | None = None
    """prompt_tokens 중 캐시로 건너뛴 몫 (Ollama prompt_eval_cached_count). prefix 캐시 효과 측정용."""
    load_ms: int | None = None
    """모델 적재 시간. 아래 둘과 함께 Ollama 만 준다 (나노초 → ms). vLLM · OpenAI 호환은 None."""
    prompt_eval_ms: int | None = None
    eval_ms: int | None = None


@dataclass(frozen=True)
class Generation:
    """생성 결과 하나. thinking 과 최종 답을 따로 담는다."""

    text: str
    """최종 답. 다음 턴 히스토리에는 이것만 넣는다."""
    model: ModelSpec
    usage: Usage
    thinking: str | None = None
    """thinking 내용. 꺼져 있거나 서버가 주지 않으면 None."""
    done_reason: DoneReason | None = None
    """멈춘 이유. thinking 이 max_tokens 를 다 쓰면 text 가 빈 채 length 다. 서버가 주지 않거나 모르는 값이면 None."""
    raw_json: str | None = None
    """서버 응답 원문 JSON. 디버그에 꼭 필요할 때만 채운다."""

    def as_message(self) -> Message:
        """히스토리에 넣을 assistant 턴 (thinking 제외)."""
        return Message("assistant", self.text)


@dataclass(frozen=True)
class StreamDelta:
    """스트리밍 중 새로 온 조각. thinking 조각과 답 조각을 구분한다."""

    kind: Literal["thinking", "text"]
    text: str


def _think(value: ParamValue) -> Think:
    # bool · 대소문자 변형은 받지 않는다 — True 와 Think.ON 처럼 한 설정이 두 표기로 기록되지 않게
    try:
        return Think(value)
    except ValueError:
        choices = [think.value for think in Think]
        raise ValueError(f"{THINK} must be one of {choices}, got {value=}") from None


def _format(value: ParamValue) -> str:
    if isinstance(value, bool):  # int 보다 먼저 — bool 은 int 의 하위 타입
        return "true" if value else "false"
    if isinstance(value, tuple):
        return "|".join(value)
    return str(value)
