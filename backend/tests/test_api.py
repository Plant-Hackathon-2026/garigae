import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app
from backend.schemas import DetectResponse

client = TestClient(app)


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
