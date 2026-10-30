# Backend (FastAPI + uv)

## 준비

```bash
cd backend
uv sync

# EasyOCR 모델(ko, en) 받기. 처음 한 번만 (약 100MB)
uv run python scripts/download_models.py
```

## 실행 방법

```bash
cd backend

# 개발 서버 실행 (코드 수정 시 자동 리로드)
uv run uvicorn backend.main:app --reload

# 또는 python으로 직접 실행
uv run python main.py
```

- API 서버 주소: `http://localhost:8000`
- 대화형 Swagger 문서: `http://localhost:8000/docs`
- Redoc 문서: `http://localhost:8000/redoc`

## 테스트

```bash
uv run pytest                 # 전체. 실제 EasyOCR 테스트는 모델이 없으면 건너뜀
uv run pytest -m "not ocr"    # 실제 OCR 제외, 빠른 테스트만
uv run pytest -m ocr          # 실제 OCR 테스트만
```

## 설정 (환경변수)

목록 값은 쉼표로 구분합니다. 예: `GARIGAE_DETECTORS=regex,rules,ner`

| 변수 | 기본값 | 설명 |
|---|---|---|
| `GARIGAE_OCR_LANGUAGES` | `ko,en` | EasyOCR 언어 |
| `GARIGAE_OCR_DEVICE` | `auto` | `auto`(CUDA → MPS → CPU 순으로 선택), `cuda`, `mps`, `cpu` |
| `GARIGAE_OCR_MODEL_DIR` | EasyOCR 기본 위치 `~/.EasyOCR/model` | 모델 저장 위치 |
| `GARIGAE_OCR_ROTATIONS` | 끔 | 90°·180° 돌아간 이미지 인식 각도. 예: `90,180,270`. 각도 수만큼 느려짐 |
| `GARIGAE_MAX_UPLOAD_BYTES` | `10485760` (10MB) | 업로드 최대 크기 |
| `GARIGAE_MAX_IMAGE_PIXELS` | `40000000` | 이미지 최대 픽셀 수 (압축 폭탄 방지) |
| `GARIGAE_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | 프론트 허용 출처 |
| `GARIGAE_DETECTORS` | `regex,rules` | 켤 감지기: `regex`, `rules`, `ner` |

## 샘플 이미지

`tests/fixtures/`에 가짜 데이터 샘플 이미지와 정답 영역(`<이름>.json`)이 있습니다. 실제 OCR 테스트(`-m ocr`)가 씁니다.

```bash
uv run python scripts/make_samples.py   # 다시 만들기 (macOS AppleSDGothicNeo 기준)
```

## OCR 처리 시간 재기

가짜 데이터 이미지로 장치와 처리 시간을 확인합니다. 위 환경변수를 그대로 따릅니다.

```bash
uv run python scripts/ocr_benchmark.py IMAGE --runs 3
GARIGAE_OCR_DEVICE=cpu uv run python scripts/ocr_benchmark.py IMAGE
```
