"""실제 EasyOCR로 도는 테스트. 모델이 없으면 건너뛴다: `uv run python scripts/download_models.py`"""

from pathlib import Path

import easyocr
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from backend.config import Settings

pytestmark = pytest.mark.ocr

# 한글 폰트 후보 (macOS, Ubuntu)
FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


@pytest.fixture(scope="module")
def reader():
    settings = Settings()
    try:
        return easyocr.Reader(
            settings.ocr_languages,
            gpu=settings.ocr_gpu,
            model_storage_directory=str(settings.ocr_model_dir) if settings.ocr_model_dir else None,
            download_enabled=False,
            verbose=False,
        )
    except FileNotFoundError as error:
        pytest.skip(f"EasyOCR 모델이 없음: {error}")


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


def test_reads_korean_and_digits(reader):
    # 가짜 데이터
    results = reader.readtext(render(["받는 분: 홍길동", "주문번호 2026-1030-0001"]))
    texts = [text for _, text, _ in results]
    assert any("홍길동" in text for text in texts), texts
    assert any("2026-1030-0001" in text for text in texts), texts
