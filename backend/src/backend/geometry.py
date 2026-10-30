"""문자 구간 → 이미지 좌표 다각형.

- 박스 하나 안의 일부 문자열(예: `연락처: 010-1234-5678`의 번호 부분)은 문자 종류별 폭 가중치로 위치를 추정한다.
  위·아래 변을 같은 비율로 잘라서 기울어진 박스도 기울기가 유지된다
- 추정이 어긋나도 덜 가리지 않도록 앞뒤로 글자 반 칸, 위아래로 높이의 10% 여백을 둔다
- 한 항목이 박스 여러 개에 걸치면 전체를 감싸는 최소 회전 직사각형(`cv2.minAreaRect`) 하나로 합친다

박스 사각형(quad)은 `(4, 2)` 배열이고, 글자 기준 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래 순서다.
화면(y가 아래로 커짐)에서 보면 시계 방향이다.
"""

import unicodedata
from collections.abc import Sequence

import cv2
import numpy as np

# 문자 종류별 폭. 한글 한 글자를 1로 본다
WIDE_WEIGHT = 1.0  # 한글, 한자, 전각 문자
ALNUM_WEIGHT = 0.65  # 계획 문서 초안 0.6. 샘플(AppleSDGothicNeo 숫자 폭 ≈ 한글의 0.67~0.72)로 보정
SPACE_WEIGHT = 0.35
PUNCT_WEIGHT = 0.4

# 앞뒤 여백 (구간 끝 글자 폭 기준 몇 칸)
SIDE_MARGIN = 0.5
# 위아래 여백 (높이 대비)
VERTICAL_MARGIN = 0.1
# EasyOCR는 글자 영역보다 박스를 조금 넓게 잡는다 (높이의 약 8%). 박스 양 끝에서 이만큼 안쪽부터 글자가 있다고 본다
BOX_END_PADDING = 0.08

# (박스 사각형, 박스 텍스트, 박스 안 문자 구간 start, end)
Piece = tuple[np.ndarray, str, int, int]


def char_weight(char: str) -> float:
    if char.isspace():
        return SPACE_WEIGHT
    if unicodedata.east_asian_width(char) in ("W", "F"):
        return WIDE_WEIGHT
    if char.isalnum():
        return ALNUM_WEIGHT
    if unicodedata.category(char)[0] in ("P", "S"):
        return PUNCT_WEIGHT
    return ALNUM_WEIGHT


def text_width(text: str) -> float:
    """문자 폭 가중치의 합. 한글 한 글자 폭이 1이다."""
    return sum(char_weight(char) for char in text)


def direction(quad: np.ndarray) -> np.ndarray:
    """글자 방향 단위 벡터 (왼쪽에서 오른쪽)."""
    vector = (quad[1] - quad[0]) + (quad[2] - quad[3])
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def side_lengths(quad: np.ndarray) -> tuple[float, float]:
    """(글자 방향 길이, 높이). 마주 보는 두 변의 평균."""
    length = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return float(length), float(height)


def clockwise(points: np.ndarray) -> np.ndarray:
    """꼭짓점을 화면 기준 시계 방향으로 정렬한다 (시작점은 정하지 않음)."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    return points[np.argsort(angles, kind="stable")]


def order_by_direction(points: np.ndarray, text_direction: np.ndarray) -> np.ndarray:
    """꼭짓점 4개를 글자 기준 왼쪽 위부터 시계 방향으로 정렬한다."""
    points = clockwise(points)
    down = np.array([-text_direction[1], text_direction[0]])
    first = int(np.argmin(points @ text_direction + points @ down))
    return np.roll(points, -first, axis=0)


def span_quad(quad: np.ndarray, text: str, start: int, end: int) -> np.ndarray:
    """박스 안 문자 구간 `text[start:end]`를 덮는 사각형 (여백 포함)."""
    weights = [char_weight(char) for char in text]
    total = sum(weights)
    length, height = side_lengths(quad)
    if total <= 0 or length <= 0:
        start_t, end_t = 0.0, 1.0
    else:
        before = sum(weights[:start])
        inside = sum(weights[start:end])
        start_f = (before - SIDE_MARGIN * weights[start]) / total
        end_f = (before + inside + SIDE_MARGIN * weights[end - 1]) / total
        # 박스 양 끝의 빈 부분을 빼고 글자가 있는 부분에서 비율을 잡는다
        pad = min(BOX_END_PADDING * height / length, 0.25)
        start_t = pad + start_f * (1 - 2 * pad)
        end_t = pad + end_f * (1 - 2 * pad)

    tl, tr, br, bl = quad
    top_start, top_end = tl + start_t * (tr - tl), tl + end_t * (tr - tl)
    bottom_start, bottom_end = bl + start_t * (br - bl), bl + end_t * (br - bl)
    grow_start = VERTICAL_MARGIN * (bottom_start - top_start)
    grow_end = VERTICAL_MARGIN * (bottom_end - top_end)
    return np.array(
        [
            top_start - grow_start,
            top_end - grow_end,
            bottom_end + grow_end,
            bottom_start + grow_start,
        ]
    )


def merge_quads(quads: Sequence[np.ndarray]) -> np.ndarray:
    """사각형 여러 개를 모두 감싸는 최소 회전 직사각형. 꼭짓점 순서는 글자 방향 기준."""
    if len(quads) == 1:
        return quads[0]
    text_direction = sum(direction(quad) for quad in quads)
    text_direction = text_direction / max(float(np.linalg.norm(text_direction)), 1e-9)
    rect = cv2.minAreaRect(np.concatenate(quads).astype(np.float32))
    return order_by_direction(cv2.boxPoints(rect).astype(np.float64), text_direction)


def span_polygon(pieces: Sequence[Piece], size: tuple[int, int]) -> np.ndarray:
    """한 항목이 걸친 박스 조각들 → 꼭짓점 4개 다각형 (시계 방향, 이미지 경계로 자름)."""
    if not pieces:
        raise ValueError("조각이 없음")
    return clip(merge_quads([span_quad(*piece) for piece in pieces]), size)


def clip(quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """꼭짓점을 이미지 경계 `[0, 너비] × [0, 높이]` 안으로 자른다."""
    width, height = size
    return np.column_stack([np.clip(quad[:, 0], 0, width), np.clip(quad[:, 1], 0, height)])


def to_points(quad: np.ndarray) -> list[tuple[float, float]]:
    """API 응답용 좌표 (소수점 한 자리)."""
    return [(round(float(x), 1), round(float(y), 1)) for x, y in quad]
