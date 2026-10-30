"""실제 EasyOCR로 도는 테스트. 모델이 없으면 건너뛴다: `uv run python scripts/download_models.py`

샘플(`tests/fixtures/`)은 `scripts/make_samples.py`로 만든 가짜 데이터다.
"""

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import polygon_mask
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from backend import image_io
from backend.config import Settings
from backend.detectors.registry import build_detectors
from backend.main import app
from backend.ocr import EasyOcrEngine
from backend.pipeline import Pipeline

pytestmark = pytest.mark.ocr

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLES = sorted(FIXTURES.glob("*.json"))
# 지금 감지기가 찾는 종류. 정답에는 다른 종류(이름, 주소, 이메일)도 들어 있다. 감지기를 채우면 늘린다
CHECKED_TYPES = {"phone"}
# 정답 영역을 이만큼 이상 덮어야 한다
MIN_COVERAGE = 0.95
# 번호만 가린다: 감지 영역이 정답 영역(글자 잉크)보다 이 배수 넘게 크면 줄 전체를 가린 것으로 본다
MAX_AREA_RATIO = 3.0

# 한글 폰트 후보 (macOS, Ubuntu)
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


@pytest.fixture(scope="module")
def engine():
    try:
        engine = EasyOcrEngine(Settings())
    except FileNotFoundError as error:
        pytest.skip(f"EasyOCR 모델이 없음: {error}")
    engine.warmup()
    return engine


@pytest.fixture(scope="module")
def pipeline(engine):
    return Pipeline(engine, build_detectors(["regex"]))


def render(lines: list[str]) -> np.ndarray:
    font_path = next((path for path in FONT_CANDIDATES if Path(path).exists()), None)
    if font_path is None:
        pytest.skip("한글 폰트가 없음")
    font = ImageFont.truetype(font_path, 40)
    image = Image.new("RGB", (800, 120 * len(lines) + 40), "white")
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(lines):
        draw.text((40, 40 + i * 120), text, fill="black", font=font)
    return np.asarray(image)


def test_reads_korean_and_digits(engine):
    # 가짜 데이터
    boxes = engine.read(render(["받는 분: 홍길동", "주문번호 2026-1030-0001"]))
    texts = [box.text for box in boxes]
    assert any("홍길동" in text for text in texts), texts
    assert any("2026-1030-0001" in text for text in texts), texts


def load(answer_path: Path) -> tuple[dict, bytes]:
    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    return answer, (FIXTURES / answer["image"]).read_bytes()


@pytest.mark.parametrize("answer_path", SAMPLES, ids=lambda path: path.stem)
def test_sample_regions_are_detected(pipeline, answer_path):
    answer, data = load(answer_path)
    image = image_io.decode(data)
    size = image.size
    assert size == (answer["width"], answer["height"])
    result = pipeline.run(image_io.to_rgb(image.pixels), types=CHECKED_TYPES)

    expected = [region for region in answer["regions"] if region["type"] in CHECKED_TYPES]
    assert expected, "확인할 정답 영역이 없음"
    detected = [(d, polygon_mask(d.polygon, size)) for d in result.detections]
    for region in expected:
        truth = polygon_mask(region["polygon"], size)
        coverage = max(((truth & mask).sum() / truth.sum() for _, mask in detected), default=0)
        assert coverage >= MIN_COVERAGE, (region["text"], coverage)
        covering = max(detected, key=lambda pair: (truth & pair[1]).sum())
        assert covering[0].type == region["type"]
        assert covering[1].sum() <= MAX_AREA_RATIO * truth.sum(), region["text"]
    # 정답이 아닌 곳을 가리지 않는다
    truths = [polygon_mask(region["polygon"], size) for region in expected]
    for detection, mask in detected:
        assert any((mask & truth).any() for truth in truths), detection.text


@pytest.mark.parametrize("answer_path", SAMPLES, ids=lambda path: path.stem)
def test_redact_auto_blurs_only_phone_numbers(pipeline, answer_path):
    answer, data = load(answer_path)
    previous = getattr(app.state, "pipeline", None)
    app.state.pipeline = pipeline
    try:
        response = TestClient(app).post(
            "/api/redact/auto", files={"file": (answer["image"], data)}, data={"types": "phone"}
        )
    finally:
        app.state.pipeline = previous
    assert response.status_code == 200
    original = image_io.decode(data)
    result = np.asarray(Image.open(io.BytesIO(response.content)).convert("RGB"))
    size = original.size
    assert (result.shape[1], result.shape[0]) == size

    diff = np.abs(result.astype(int) - image_io.to_rgb(original.pixels).astype(int)).max(axis=2)
    for region in answer["regions"]:
        if region["type"] in CHECKED_TYPES:
            # 글자 획이 지워질 만큼 바뀐다
            truth = polygon_mask(region["polygon"], size)
            assert (diff[truth] > 30).mean() > 0.2, region["text"]
    if original.format == "PNG":
        # 감지 영역 바깥(경계 2픽셀 제외)은 픽셀이 그대로다
        detections = pipeline.run(image_io.to_rgb(original.pixels), types=CHECKED_TYPES).detections
        blurred = np.zeros(diff.shape, bool)
        for detection in detections:
            blurred |= polygon_mask(detection.polygon, size)
        near = cv2.dilate(blurred.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        assert not diff[~near].any()
