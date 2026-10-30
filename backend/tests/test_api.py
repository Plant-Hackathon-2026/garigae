import importlib
import io
import json
import time

import numpy as np
import pytest
from fakes import FakeOcrEngine, make_box, polygon_mask
from fastapi.testclient import TestClient
from PIL import Image

from backend.detectors.registry import build_detectors
from backend.main import app
from backend.pipeline import Pipeline
from backend.schemas import DetectResponse

# `backend/__init__.py`의 main() 함수와 이름이 같아서 모듈을 직접 가져온다
main_module = importlib.import_module("backend.main")
client = TestClient(app)


@pytest.fixture(autouse=True)
def fake_pipeline():
    """OCR 모델 대신 가짜 엔진을 넣어 둔다. `TestClient`를 with 없이 쓰면 lifespan(모델 로딩)이 돌지 않는다."""
    # app.state는 monkeypatch.setattr로 되돌리면 속성이 지워져서 직접 저장·복원한다
    previous = getattr(app.state, "pipeline", None)
    pipeline = Pipeline(FakeOcrEngine(), build_detectors(["regex"]))
    app.state.pipeline = pipeline
    yield pipeline
    app.state.pipeline = previous


def image_bytes(size=(640, 480), fmt="PNG", orientation: int | None = None) -> bytes:
    image = Image.new("RGB", size, "white")
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, exif=exif)
    return buffer.getvalue()


def upload(data: bytes, name="sample.png", media_type="image/png"):
    return {"file": (name, data, media_type)}


def test_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert set(body) == {"status", "ocr_ready", "device"}


def test_pii_types():
    response = client.get("/api/pii-types")
    assert response.status_code == 200
    labels = {item["type"]: item["label"] for item in response.json()}
    assert len(labels) == 10
    assert labels["phone"] == "전화번호"


def test_detect_returns_contract():
    response = client.post("/api/detect", files=upload(image_bytes((640, 480))))
    assert response.status_code == 200
    body = DetectResponse.model_validate(response.json())
    assert (body.image.width, body.image.height, body.image.format) == (640, 480, "png")
    for item in [*body.detections, *body.text_regions]:
        assert len(item.polygon) == 4


def test_detect_size_follows_exif_rotation():
    # 6: 90° 회전. 세로로 찍은 폰 사진처럼 가로·세로가 바뀌어야 한다
    data = image_bytes((640, 480), fmt="JPEG", orientation=6)
    response = client.post("/api/detect", files=upload(data, "photo.jpg", "image/jpeg"))
    assert response.status_code == 200
    image = response.json()["image"]
    assert (image["width"], image["height"], image["format"]) == (480, 640, "jpeg")


def test_detect_rejects_non_image():
    response = client.post("/api/detect", files=upload(b"not an image", "a.png"))
    assert response.status_code == 400


def test_detect_rejects_unsupported_format():
    data = image_bytes(fmt="GIF")
    response = client.post("/api/detect", files=upload(data, "a.gif", "image/gif"))
    assert response.status_code == 415


REGIONS = [
    {"polygon": [[10, 10], [100, 10], [100, 40], [10, 40]]},
    {"polygon": [[20.5, 60], [120, 70.25], [118, 100], [18, 90]]},
]


@pytest.mark.parametrize("mode", ["blur", "pixelate", "box"])
def test_redact_returns_image_of_same_size(mode):
    response = client.post(
        "/api/redact",
        files=upload(image_bytes((640, 480))),
        data={"regions": json.dumps(REGIONS), "mode": mode},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert Image.open(io.BytesIO(response.content)).size == (640, 480)


@pytest.mark.parametrize(
    "regions",
    [
        "not json",
        json.dumps([{"polygon": [[10, 10], [100, 10], [100, 40]]}]),  # 꼭짓점 3개
        json.dumps([{"bbox": [10, 10, 100, 40]}]),
        json.dumps({"polygon": [[10, 10], [100, 10], [100, 40], [10, 40]]}),  # 목록이 아님
    ],
)
def test_redact_rejects_bad_regions(regions):
    response = client.post(
        "/api/redact", files=upload(image_bytes()), data={"regions": regions}
    )
    assert response.status_code == 400


def test_redact_auto_returns_image():
    response = client.post(
        "/api/redact/auto",
        files=upload(image_bytes((320, 240), fmt="JPEG"), "a.jpg", "image/jpeg"),
        data={"types": "phone,email"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert Image.open(io.BytesIO(response.content)).size == (320, 240)


def test_redact_auto_rejects_unknown_type():
    response = client.post(
        "/api/redact/auto", files=upload(image_bytes()), data={"types": "phone,face"}
    )
    assert response.status_code == 400


def test_cors_allows_frontend_origin():
    response = client.options(
        "/api/health",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_health_reports_ocr_state():
    assert client.get("/api/health").json() == {
        "status": "healthy",
        "ocr_ready": True,
        "device": "fake",
    }
    app.state.pipeline = None
    assert client.get("/api/health").json() == {
        "status": "healthy",
        "ocr_ready": False,
        "device": None,
    }


def test_redact_auto_before_ocr_is_ready():
    app.state.pipeline = None
    response = client.post("/api/redact/auto", files=upload(image_bytes()))
    assert response.status_code == 503


def noise_png(size=(640, 240)) -> tuple[np.ndarray, bytes]:
    pixels = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return pixels, buffer.getvalue()


def test_redact_auto_blurs_only_detected_phone(fake_pipeline):
    fake_pipeline.engine.boxes = [
        make_box("받는 분: 김가상", 40, 40, 300, 44),
        make_box("연락처: 010-1234-5678", 40, 120, 440, 44),
    ]
    pixels, data = noise_png()
    response = client.post("/api/redact/auto", files=upload(data))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    result = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))

    [detection] = fake_pipeline.run(pixels).detections
    assert detection.text == "010-1234-5678"
    inside = polygon_mask(detection.polygon, (640, 240))
    diff = (result != pixels).any(axis=2)
    assert diff[inside].mean() > 0.95
    # 커널이 닿는 경계 1~2픽셀을 빼면 다각형 바깥은 그대로다 (라벨 `연락처:`, 다른 줄 포함)
    grown = np.zeros_like(inside)
    ys, xs = np.nonzero(inside)
    grown[max(ys.min() - 2, 0) : ys.max() + 3, max(xs.min() - 2, 0) : xs.max() + 3] = True
    assert not diff[~grown].any()
    assert not diff[120:164, 40:120].any()


def test_redact_auto_types_filter(fake_pipeline):
    fake_pipeline.engine.boxes = [make_box("연락처: 010-1234-5678", 40, 120, 440, 44)]
    pixels, data = noise_png()
    response = client.post("/api/redact/auto", files=upload(data), data={"types": "email"})
    assert response.status_code == 200
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(response.content))), pixels)


def test_redact_auto_rotates_and_strips_metadata():
    # 6: 90° 회전. 결과는 바로 세운 상태이고 EXIF가 없다
    data = image_bytes((640, 480), fmt="JPEG", orientation=6)
    response = client.post("/api/redact/auto", files=upload(data, "photo.jpg", "image/jpeg"))
    assert response.status_code == 200
    image = Image.open(io.BytesIO(response.content))
    assert image.format == "JPEG"
    assert image.size == (480, 640)
    assert not image.getexif()
    assert "exif" not in image.info


@pytest.mark.parametrize("mode", ["pixelate", "box"])
def test_redact_auto_modes_not_ready_yet(mode):
    response = client.post("/api/redact/auto", files=upload(image_bytes()), data={"mode": mode})
    assert response.status_code == 501


def test_lifespan_keeps_injected_pipeline(monkeypatch):
    calls = []
    monkeypatch.setattr(main_module, "load_pipeline", lambda *args: calls.append(args))
    with TestClient(app) as lifespan_client:
        assert lifespan_client.get("/api/health").json()["device"] == "fake"
    assert calls == []


def test_lifespan_loads_ocr_in_background(monkeypatch):
    class LoadedEngine(FakeOcrEngine):
        device = "loaded"

        def __init__(self, settings):
            super().__init__()

        def warmup(self):
            pass

    monkeypatch.setattr(main_module, "EasyOcrEngine", LoadedEngine)
    app.state.pipeline = None
    with TestClient(app) as lifespan_client:
        deadline = time.monotonic() + 5
        while not lifespan_client.get("/api/health").json()["ocr_ready"]:
            assert time.monotonic() < deadline, "OCR 로딩이 끝나지 않음"
            time.sleep(0.01)
        assert lifespan_client.get("/api/health").json()["device"] == "loaded"
        response = lifespan_client.post("/api/redact/auto", files=upload(image_bytes()))
        assert response.status_code == 200
