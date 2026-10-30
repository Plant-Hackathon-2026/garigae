"""테스트용 가짜 OCR 엔진과 박스 만들기. torch 없이 파이프라인·API를 빠르게 테스트한다."""

import math

import numpy as np

from backend.ocr import OcrBox


class FakeOcrEngine:
    """이미지와 상관없이 정해 둔 박스를 돌려준다."""

    device = "fake"

    def __init__(self, boxes: list[OcrBox] | None = None) -> None:
        self.boxes = list(boxes or [])
        self.calls = 0

    def read(self, image: np.ndarray) -> list[OcrBox]:
        assert image.ndim == 3 and image.shape[2] == 3 and image.dtype == np.uint8
        self.calls += 1
        return list(self.boxes)


def make_box(
    text: str, x: float, y: float, width: float, height: float, confidence: float = 0.95
) -> OcrBox:
    """왼쪽 위 (x, y), 크기 width × height인 수평 글자 박스."""
    return tilted_box(text, (x + width / 2, y + height / 2), width, height, 0.0, confidence)


def tilted_box(
    text: str,
    center: tuple[float, float],
    width: float,
    height: float,
    angle: float,
    confidence: float = 0.95,
) -> OcrBox:
    """중심 `center`, 글자 방향 길이 width, 높이 height인 박스를 화면에서 반시계로 `angle`° 돌린 것.

    꼭짓점은 EasyOCR처럼 x+y가 가장 작은 점부터 시계 방향으로 준다.
    """
    corners = rotated_rect(center[0], center[1], width, height, angle)
    first = int(np.argmin(corners.sum(axis=1)))
    points = np.roll(corners, -first, axis=0)
    return OcrBox(
        points=tuple((float(px), float(py)) for px, py in points), text=text, confidence=confidence
    )


def rotated_rect(cx: float, cy: float, width: float, height: float, angle: float) -> np.ndarray:
    """글자 기준 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래. 화면에서 반시계로 `angle`° 돌린다."""
    theta = math.radians(angle)
    # 화면 좌표(y 아래)에서 반시계 회전: 글자 방향 (cos, -sin), 아래 방향 (sin, cos)
    u = np.array([math.cos(theta), -math.sin(theta)])
    v = np.array([math.sin(theta), math.cos(theta)])
    c = np.array([cx, cy])
    hw, hh = width / 2, height / 2
    return np.array([c - hw * u - hh * v, c + hw * u - hh * v, c + hw * u + hh * v, c - hw * u + hh * v])


def along(origin: tuple[float, float], angle: float, distance: float) -> tuple[float, float]:
    """`origin`에서 글자 방향(반시계 `angle`°)으로 `distance`만큼 간 점."""
    theta = math.radians(angle)
    return origin[0] + distance * math.cos(theta), origin[1] - distance * math.sin(theta)


def signed_area(polygon) -> float:
    """화면 좌표(y 아래)에서 시계 방향이면 양수."""
    points = np.asarray(polygon, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return float((x * np.roll(y, -1) - np.roll(x, -1) * y).sum() / 2)


def polygon_mask(polygon, size: tuple[int, int]) -> np.ndarray:
    """다각형 안쪽 픽셀 마스크 (높이, 너비) bool."""
    import cv2

    mask = np.zeros((size[1], size[0]), np.uint8)
    points = np.round(np.asarray(polygon, dtype=np.float64) * 16).astype(np.int32)
    cv2.fillPoly(mask, [points], 1, shift=4)
    return mask.astype(bool)
