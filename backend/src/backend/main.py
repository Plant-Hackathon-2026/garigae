import logging
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api import router
from backend.config import Settings, get_settings
from backend.detectors.registry import build_detectors
from backend.ocr import EasyOcrEngine
from backend.pipeline import Pipeline

settings = get_settings()
logger = logging.getLogger(__name__)

# backend.* 로그(처리 시간, 개수)를 uvicorn 로그와 같이 보이게 한다
_backend_logger = logging.getLogger("backend")
if not _backend_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s:     %(name)s - %(message)s"))
    _backend_logger.addHandler(_handler)
    _backend_logger.setLevel(logging.INFO)


def load_pipeline(app: FastAPI, settings: Settings) -> None:
    """OCR 모델 로딩 + 워밍업, 감지기 준비. 끝나면 `app.state.pipeline`에 넣는다."""
    started = time.perf_counter()
    try:
        engine = EasyOcrEngine(settings)
        engine.warmup()
        detectors = build_detectors(settings.detectors)
    except Exception:
        logger.exception("OCR 모델 로딩 실패. 요청은 503으로 응답한다")
        return
    app.state.pipeline = Pipeline(engine, detectors)
    logger.info(
        "OCR 준비 완료: device=%s, detectors=%s, %.1fs",
        engine.device,
        ",".join(detector.name for detector in detectors),
        time.perf_counter() - started,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로딩(수 초)은 백그라운드에서 한다. 그동안 /api/health는 ocr_ready=false, 처리 요청은 503.
    # 테스트처럼 파이프라인을 미리 넣어 두면 불러오지 않는다
    if getattr(app.state, "pipeline", None) is None:
        threading.Thread(
            target=load_pipeline, args=(app, settings), name="ocr-loader", daemon=True
        ).start()
    yield


app = FastAPI(
    title="Garigae Backend API",
    description="FastAPI backend with uv",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.pipeline = None

# 프론트엔드와의 통신을 위한 CORS 설정 (허용 출처: GARIGAE_CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI backend!", "status": "ok"}
