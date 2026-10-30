# Backend (FastAPI + uv)

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
