"""이미지 디코딩·인코딩.

- 디코딩: EXIF 회전 정보(90° 단위 방향 값)를 픽셀에 적용해 바로 세운다. 좌표는 모두 이 상태 기준이다
- 인코딩: 픽셀 배열로 이미지를 새로 만들어 저장한다. EXIF(GPS, 촬영 정보, 원본 썸네일), XMP,
  PNG 텍스트 정보 같은 메타데이터가 따라오지 않는다

TODO(#9): 업로드 크기·픽셀 수 검사, 색상 프로필(ICC) 유지, 관련 테스트
"""

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

FORMATS = ("JPEG", "PNG", "WEBP")
MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_SAVE_OPTIONS: dict[str, dict] = {"JPEG": {"quality": 95}, "PNG": {}, "WEBP": {"quality": 95}}


class ImageError(Exception):
    status_code = 400


class InvalidImageError(ImageError):
    status_code = 400


class UnsupportedFormatError(ImageError):
    status_code = 415


@dataclass(frozen=True, eq=False)
class DecodedImage:
    # (높이, 너비, 3) RGB 또는 투명도가 있으면 (높이, 너비, 4) RGBA, uint8. EXIF 회전 정보를 적용한 상태
    pixels: np.ndarray
    # JPEG, PNG, WEBP
    format: str

    @property
    def size(self) -> tuple[int, int]:
        """(너비, 높이)."""
        return self.pixels.shape[1], self.pixels.shape[0]


def decode(data: bytes) -> DecodedImage:
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise InvalidImageError("이미지를 열 수 없음") from None
    # 확장자나 Content-Type이 아니라 실제 파일 내용으로 판단한 형식
    image_format = image.format
    if image_format not in FORMATS:
        raise UnsupportedFormatError("JPEG/PNG/WEBP만 지원")
    try:
        image = ImageOps.exif_transpose(image)
        has_alpha = image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info
        pixels = np.asarray(image.convert("RGBA" if has_alpha else "RGB"))
    except (OSError, ValueError):
        raise InvalidImageError("이미지를 열 수 없음") from None
    return DecodedImage(pixels=pixels, format=image_format)


def encode(pixels: np.ndarray, image_format: str) -> bytes:
    """입력과 같은 형식으로 저장한다. 메타데이터는 넣지 않는다."""
    if image_format == "JPEG" and pixels.ndim == 3 and pixels.shape[2] == 4:
        pixels = pixels[:, :, :3]
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(pixels)).save(
        buffer, format=image_format, **_SAVE_OPTIONS[image_format]
    )
    return buffer.getvalue()


def to_rgb(pixels: np.ndarray) -> np.ndarray:
    """OCR 입력용 RGB. 투명한 부분은 흰 배경 위에 합성한다 (투명 배경의 검은 글자가 검게 묻히지 않게)."""
    if pixels.shape[2] == 3:
        return pixels
    alpha = pixels[:, :, 3:4].astype(np.float32) / 255
    rgb = pixels[:, :, :3].astype(np.float32) * alpha + 255 * (1 - alpha)
    return np.round(rgb).astype(np.uint8)
