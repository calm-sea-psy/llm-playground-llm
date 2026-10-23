import pytest

from playground_evalkit import ResultRecord
from playground_llm import THINK, Generation, Message, ModelSpec, ParamValue, Think, Usage

SPEC = ModelSpec("ollama", "gemma4:12b", {"temperature": 0.1, "stop": ("<eos>", "\n\n")})


def test_spec_id_sorts_params_and_formats_values():
    spec = ModelSpec(
        "ollama", "gemma4:12b", {"think": Think.OFF, "temperature": 0.1, "num_ctx": 8192}
    )

    assert spec.spec_id == "ollama/gemma4:12b?num_ctx=8192,temperature=0.1,think=off"
    assert SPEC.spec_id == "ollama/gemma4:12b?stop=<eos>|\n\n,temperature=0.1"
    assert ModelSpec("ollama", "gemma3:4b").spec_id == "ollama/gemma3:4b"


def test_think_on_and_off_are_different_specs():
    off = SPEC.with_params(think=Think.OFF)
    on = SPEC.with_params(think=Think.ON)

    assert (off.think, on.think, SPEC.think) == (Think.OFF, Think.ON, Think.OFF)
    assert off.spec_id != on.spec_id
    assert off != on
    assert on.params[THINK] is Think.ON
    assert SPEC.params.get(THINK) is None  # 원래 spec 은 그대로


def test_think_string_becomes_enum_and_level_counts_as_enabled():
    low = SPEC.with_params(think="low")

    assert low.params[THINK] is Think.LOW
    assert low == SPEC.with_params(think=Think.LOW)
    assert low.spec_id.endswith(",think=low")
    assert [think.enabled for think in Think] == [False, True, True, True, True]


@pytest.mark.parametrize("value", [True, False, "Low", "true", 1])
def test_spec_rejects_think_outside_enum(value: ParamValue):
    # bool 을 받으면 True 와 Think.ON 이 같은 설정의 두 표기로 기록된다
    with pytest.raises(ValueError, match="think must be one of"):
        SPEC.with_params(think=value)


def test_spec_is_hashable_and_equal_by_value():
    same = ModelSpec("ollama", "gemma4:12b", {"stop": ("<eos>", "\n\n"), "temperature": 0.1})

    assert same == SPEC
    assert hash(same) == hash(SPEC)
    assert {SPEC: "profile"}[same] == "profile"


def test_spec_params_cannot_change_after_creation():
    params = {"temperature": 0.1}
    spec = ModelSpec("ollama", "m", params)
    params["temperature"] = 0.9

    assert spec.params["temperature"] == 0.1
    with pytest.raises(TypeError):
        spec.params["temperature"] = 0.5  # type: ignore[index]  # 읽기 전용 확인


def test_spec_rejects_empty_names():
    with pytest.raises(ValueError, match="must not be empty"):
        ModelSpec("", "m")


def test_message_rejects_unknown_role():
    with pytest.raises(ValueError, match="role must be one of"):
        Message("tool", "x")  # type: ignore[arg-type]  # 잘못된 값 확인


def test_generation_history_message_drops_thinking():
    generation = Generation(
        text="서울",
        model=SPEC,
        usage=Usage(prompt_tokens=3, completion_tokens=40, total_ms=900),
        thinking="한양은 조선의 수도였으므로…",
    )

    assert generation.as_message() == Message("assistant", "서울")
    assert generation.usage.thinking_tokens is None  # Ollama 는 따로 주지 않는다


def test_spec_params_fit_evalkit_result_record():
    record = ResultRecord(
        run_id="r", spec_id=SPEC.spec_id, item_id="q1", scores={"em": 1.0}, spec=SPEC.params
    )

    assert record.spec["stop"] == ("<eos>", "\n\n")
