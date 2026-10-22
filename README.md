# llm-playground-llm

llm-playground 프로젝트들이 함께 쓰는 LLM · 평가 모듈입니다. uv workspace 로 패키지 두 개를 둡니다.

| 패키지 | 내용 | 상태 |
|---|---|---|
| `playground_evalkit` (`packages/evalkit`) | 검색 · 생성 지표, KorQuAD 로더와 정답 매칭, 결과 기록 · 리포트. **의존 없음** | 사용 중 |
| `playground_llm` (`packages/llm`) | LLM 클라이언트 · 프롬프트 | 예정 |

## 설치

```toml
[tool.uv.sources]
playground-evalkit = { git = "https://github.com/calm-sea-psy/llm-playground-llm", tag = "evalkit-v0.1.0", subdirectory = "packages/evalkit" }
```

이 레포와 함께 고칠 때는 같은 부모 폴더에 나란히 clone 하고 잠깐 path 로 바꿔 씁니다: `{ path = "../llm-playground-llm/packages/evalkit", editable = true }`.

0.x 동안은 깨는 변경이 있을 수 있습니다. 태그는 패키지별 접두사(`evalkit-v…`, `llm-v…`)를 씁니다.

## 개발

```bash
uv sync --all-packages
uv run pytest
uv run ruff check
```

## 구조

```
packages/evalkit/   playground_evalkit
packages/llm/       playground_llm (예정)
tests/              패키지별 테스트 (tests/evalkit/ …)
bench/              모델 벤치 (bench/data/ 는 git 제외)
reports/            벤치 결과 — 집계 지표만 커밋
models/             선정 결과 yaml
.ruff/              공유 ruff 설정 복사본 (직접 고치지 않음)
```
