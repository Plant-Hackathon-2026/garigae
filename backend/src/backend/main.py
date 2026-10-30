from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Garigae Backend API",
    description="FastAPI backend with uv",
    version="0.1.0",
)

# 프론트엔드와의 통신을 위한 CORS 설정 (허용 출처: GARIGAE_CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!", "status": "ok"}


@app.get("/api/health")
def health_check():
    return {"status": "healthy"}
