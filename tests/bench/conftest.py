import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

BENCH_SCRIPT = Path(__file__).parents[2] / "bench" / "f4_bench.py"


@pytest.fixture(scope="session")
def f4_bench() -> ModuleType:
    # bench/ 는 패키지가 아니라 실행 스크립트 폴더라 파일 경로로 읽는다. 읽기만 하고 Ollama 는 부르지 않는다
    spec = importlib.util.spec_from_file_location("f4_bench", BENCH_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["f4_bench"] = module  # dataclass 가 정의 모듈을 sys.modules 에서 찾는다
    spec.loader.exec_module(module)
    return module
