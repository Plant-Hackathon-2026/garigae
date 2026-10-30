from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Garigae Backend API",
    description="FastAPI backend with uv",
    version="0.1.0",
)

# Vite 프론트엔드와의 통신을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
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
