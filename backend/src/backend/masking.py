"""영역 가리기. 다각형 안쪽 픽셀만 바꾸고 바깥은 그대로 둔다.

TODO(#9): pixelate, box 모드
"""

import math
from collections.abc import Sequence

import cv2
import numpy as np

PolygonLike = Sequence[Sequence[float]]


def blur(image: np.ndarray, polygons: Sequence[PolygonLike]) -> np.ndarray:
    """다각형마다 안쪽을 흐리게 한 사본을 돌려준다 (원본은 그대로).

    커널 크기는 다각형의 짧은 변(기울어진 글자면 글자 높이 방향) 이상으로 잡아 글자를 알아볼 수 없게 한다.
    수평 외접 사각형의 높이는 기울어진 글자에서 실제 글자보다 크므로 쓰지 않는다.
    """
    result = image.copy()
    height, width = image.shape[:2]
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
        kernel = kernel_size(points)
        # 커널이 닿는 범위까지 잘라서 흐리게 한다. 경계 픽셀도 주변 픽셀로 흐려지게 커널만큼 넓힌다
        x0 = max(math.floor(points[:, 0].min()) - kernel, 0)
        y0 = max(math.floor(points[:, 1].min()) - kernel, 0)
        x1 = min(math.ceil(points[:, 0].max()) + kernel + 1, width)
        y1 = min(math.ceil(points[:, 1].max()) + kernel + 1, height)
        if x1 <= x0 or y1 <= y0:
            continue
        region = result[y0:y1, x0:x1]
        mask = np.zeros(region.shape[:2], np.uint8)
        # 4비트 소수점 좌표로 채운다 (shift=4)
        fixed = np.round((points - (x0, y0)) * 16).astype(np.int32)
        cv2.fillPoly(mask, [fixed], 255, lineType=cv2.LINE_8, shift=4)
        if not mask.any():
            continue
        blurred = cv2.GaussianBlur(
            region, (kernel, kernel), sigmaX=kernel / 4, borderType=cv2.BORDER_REPLICATE
        )
        inside = mask.astype(bool)
        region[inside] = blurred[inside]
    return result


def kernel_size(points: np.ndarray) -> int:
    """다각형의 짧은 변 이상인 가장 작은 홀수 (최소 3)."""
    edges = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    if len(edges) == 4:
        short = min((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2)
    else:
        short = float(edges.min())
    size = max(math.ceil(short), 3)
    return size if size % 2 else size + 1
