"""이미지 입출력 최소 테스트. 크기 제한·ICC 유지 등은 #9에서 채운다."""

import io

import numpy as np
import pytest
from PIL import Image

from backend import image_io


def encoded(size=(64, 48), fmt="PNG", mode="RGB", orientation: int | None = None) -> bytes:
    image = Image.new(mode, size, "white")
    image.putpixel((0, 0), (255, 0, 0) if mode == "RGB" else (255, 0, 0, 128))
    exif = Image.Exif()
    if orientation is not None:
        exif[0x0112] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, exif=exif)
    return buffer.getvalue()


def test_applies_exif_rotation():
    # 6: 시계 방향 90°로 돌려 봐야 하는 사진. 가로·세로가 바뀌고 왼쪽 위 빨간 점이 오른쪽 위로 간다
    decoded = image_io.decode(encoded((64, 48), "PNG", orientation=6))
    assert decoded.format == "PNG"
    assert decoded.size == (48, 64)
    assert tuple(decoded.pixels[0, -1]) == (255, 0, 0)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_encode_keeps_format_and_drops_metadata(fmt):
    decoded = image_io.decode(encoded(fmt=fmt, orientation=1))
    image = Image.open(io.BytesIO(image_io.encode(decoded.pixels, decoded.format)))
    assert image.format == fmt
    assert image.size == (64, 48)
    assert not image.getexif()
    assert "exif" not in image.info


def test_transparency_is_kept_and_flattened_for_ocr():
    decoded = image_io.decode(encoded(mode="RGBA"))
    assert decoded.pixels.shape == (48, 64, 4)
    rgb = image_io.to_rgb(decoded.pixels)
    assert rgb.shape == (48, 64, 3)
    # 반투명 빨강은 흰 배경 위에 합성
    assert tuple(rgb[0, 0]) == (255, 127, 127)
    again = Image.open(io.BytesIO(image_io.encode(decoded.pixels, "PNG")))
    assert again.mode == "RGBA"


def test_rejects_non_image_and_unsupported_format():
    with pytest.raises(image_io.InvalidImageError):
        image_io.decode(b"not an image")
    with pytest.raises(image_io.UnsupportedFormatError):
        image_io.decode(encoded(fmt="GIF"))


def test_to_rgb_passes_rgb_through():
    pixels = np.zeros((2, 2, 3), np.uint8)
    assert image_io.to_rgb(pixels) is pixels
