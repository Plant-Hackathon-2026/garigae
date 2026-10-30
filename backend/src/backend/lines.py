"""OCR 글자 박스를 줄 단위로 묶어 줄 텍스트를 만든다.

EasyOCR는 띄어쓰기나 간격에 따라 `010 1234 5678`을 박스 여러 개로 쪼개서 준다. 박스마다 감지하면 놓치므로
같은 줄의 박스를 이어 붙인 줄 텍스트를 감지기에 넘긴다.

- 글자 방향이 비슷하고, 그 방향에 수직인 축으로 위치가 겹치는 박스를 한 줄로 묶는다
- 세로 위치만으로 묶지 않는다. EasyOCR는 약 5.7° 넘게 기울어진 박스를 옆 박스와 합치지 않고 따로 주는데,
  기울어진 줄에서는 단어마다 세로 위치가 달라서 세로 위치로는 한 줄로 묶이지 않는다
- 줄 안에서는 글자 방향을 따라 정렬해 공백 하나로 이어 붙인다
- 줄 텍스트의 문자마다 몇 번째 박스의 몇 번째 문자인지 기록해 둔다. 좌표 복원은 geometry가 맡는다
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np

from backend import geometry
from backend.detectors.base import TextLine
from backend.ocr import OcrBox

# 같은 줄로 보는 조건
ANGLE_TOLERANCE_DEG = 10.0  # 글자 방향 차이
OFFSET_RATIO = 0.5  # 글자 방향에 수직인 축으로 잰 중심 거리 / 평균 높이
MIN_HEIGHT_RATIO = 0.5  # 작은 높이 / 큰 높이

# 박스 가로세로 비가 1에 이만큼 가깝거나, 텍스트로 예상한 비가 1에 이만큼 가까우면 모양만으로는
# 글자 방향을 정하지 않는다 (정사각형에 가까운 한두 글자짜리 박스)
AMBIGUOUS_RATIO = 1.3
# 축에 나란한 박스가 이보다 정사각형에 가까우면 기울어진 단어일 수 있어 방향을 믿지 않는다.
# EasyOCR(CRAFT)는 단어 영역의 최소 회전 직사각형이 정사각형에 가까우면(비 1.1 이내) 기울어진 단어라도
# 축에 나란한 외접 사각형으로 바꿔 준다. 20° 기울어진 두 글자 단어가 1.3배 정도로 나온다
LOOSE_ASPECT = 1.5
# 폭 1(한글 한 글자)인 텍스트를 감싼 박스의 가로/세로 비. 박스 높이가 글자보다 조금 크다
BOX_ASPECT_PER_WIDTH = 0.8


@dataclass(frozen=True, eq=False)
class TextBox:
    """글자 방향을 정한 OCR 박스."""

    text: str
    confidence: float
    # (4, 2) 글자 기준 왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 (화면에서 시계 방향)
    quad: np.ndarray


# (줄의 boxes 안 위치, 박스 텍스트 안 문자 위치)
CharRef = tuple[int, int]


@dataclass(frozen=True, eq=False)
class LineLayout:
    line: TextLine
    # 글자 방향 순서
    boxes: tuple[TextBox, ...]
    # `line.text`의 문자마다 어느 박스의 몇 번째 문자인지. 박스 사이에 넣은 공백은 None
    char_map: tuple[CharRef | None, ...]

    def pieces(self, start: int, end: int) -> list[geometry.Piece]:
        """줄 텍스트의 문자 구간 `[start, end)`가 걸친 박스별 조각 (박스 사각형, 박스 텍스트, 박스 안 start, end)."""
        ranges: dict[int, tuple[int, int]] = {}
        for ref in self.char_map[start:end]:
            if ref is None:
                continue
            pos, char = ref
            first, last = ranges.get(pos, (char, char))
            ranges[pos] = (min(first, char), max(last, char))
        return [
            (self.boxes[pos].quad, self.boxes[pos].text, first, last + 1)
            for pos, (first, last) in sorted(ranges.items())
        ]

    def boxes_in(self, start: int, end: int) -> list[TextBox]:
        """문자 구간 `[start, end)`가 걸친 박스."""
        positions = {ref[0] for ref in self.char_map[start:end] if ref is not None}
        return [self.boxes[pos] for pos in sorted(positions)]


def build_lines(boxes: Sequence[OcrBox]) -> list[LineLayout]:
    """박스를 줄로 묶는다. 결과는 위에서 아래 순서이고 `line.id`는 그 순서(0부터)다."""
    items = [_Item.from_box(box) for box in boxes if box.text.strip()]
    lines = [_make_line(group) for group in _group(items)]
    if not lines:
        return []

    # 위에서 아래로: 줄 방향에 수직인 축(줄 길이로 가중 평균)으로 정렬. 같으면 글자 방향 순서
    down = sum(line.length * _down(line.direction) for line in lines)
    down = down / max(float(np.linalg.norm(down)), 1e-9)
    across = np.array([down[1], -down[0]])
    lines.sort(key=lambda line: (float(line.center @ down), float(line.center @ across)))

    layouts = []
    for line_id, line in enumerate(lines):
        texts, char_map = [], []
        for pos, box in enumerate(line.boxes):
            if pos:
                texts.append(" ")
                char_map.append(None)
            texts.append(box.text)
            char_map.extend((pos, i) for i in range(len(box.text)))
        layouts.append(
            LineLayout(
                line=TextLine(id=line_id, text="".join(texts)),
                boxes=tuple(line.boxes),
                char_map=tuple(char_map),
            )
        )
    return layouts


def orient(points: np.ndarray, text: str, prefer: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """EasyOCR 꼭짓점을 글자 기준 순서(왼쪽 위부터 시계 방향)로 바꾼다. (사각형, 방향이 애매한지)

    EasyOCR는 x+y가 가장 작은 점부터 시계 방향으로 주므로 45° 넘게 기울면 첫 점이 글자 기준 왼쪽 위가 아니다.
    그래서 긴 변을 글자 방향으로 본다. 다만 `1`처럼 한두 글자짜리 박스는 세로가 더 길 수 있으므로
    텍스트 길이로 예상한 가로세로 비와 박스 모양이 맞는 쪽을 고른다. 정사각형에 가까워 정할 수 없으면
    `prefer` 방향(같은 줄의 다른 박스), 없으면 수평에 가까운 쪽을 고르고 애매하다고 표시한다.
    글자는 왼쪽에서 오른쪽으로 읽는다고 본다 (정확히 세로인 박스는 위에서 아래).
    """
    quad = geometry.clockwise(np.asarray(points, dtype=np.float64).reshape(4, 2))
    edges = np.roll(quad, -1, axis=0) - quad
    lengths = np.linalg.norm(edges, axis=1)
    units = edges / np.maximum(lengths, 1e-9)[:, None]

    # 0·2번 변 쪽이 1·3번 변 쪽보다 몇 배 긴지, 텍스트로 예상한 가로/세로 비
    box_ratio = max(lengths[0] + lengths[2], 1e-9) / max(lengths[1] + lengths[3], 1e-9)
    expected = max(geometry.text_width(text) * BOX_ASPECT_PER_WIDTH, 1e-9)
    limit = math.log(AMBIGUOUS_RATIO)
    ambiguous = abs(math.log(box_ratio)) < limit or abs(math.log(expected)) < limit

    if ambiguous:
        target = prefer if prefer is not None else np.array([1.0, 0.0])
        first = int(np.argmax(units @ target))
    else:
        # 텍스트가 가로로 길면(expected > 1) 긴 변이, 세로로 길면 짧은 변이 글자 방향
        first = 0 if (box_ratio > 1) == (expected > 1) else 1
        if not _reads_forward(units[first]):
            first += 2
    return np.roll(quad, -first, axis=0), ambiguous


def _reads_forward(unit: np.ndarray) -> bool:
    return bool(unit[0] > 1e-6 or (abs(unit[0]) <= 1e-6 and unit[1] >= 0))


def _down(unit: np.ndarray) -> np.ndarray:
    """글자 방향에 수직이고 글자 기준 아래쪽을 가리키는 벡터."""
    return np.array([-unit[1], unit[0]])


@dataclass(eq=False)
class _Item:
    box: OcrBox
    quad: np.ndarray
    # 박스 모양으로는 글자 방향을 알 수 없다. 같은 줄로 볼지 따질 때 상대 박스의 방향을 따른다
    loose: bool
    cx: float
    cy: float
    ux: float
    uy: float
    height: float

    @classmethod
    def from_box(cls, box: OcrBox) -> "_Item":
        quad, ambiguous = orient(np.asarray(box.points, dtype=np.float64), box.text)
        length, height = geometry.side_lengths(quad)
        unit = geometry.direction(quad)
        axis_aligned = min(abs(unit[0]), abs(unit[1])) < 1e-6
        squarish = max(length, height) < LOOSE_ASPECT * min(length, height)
        loose = ambiguous or (axis_aligned and squarish)
        cx, cy = quad.mean(axis=0)
        return cls(
            box=box,
            quad=quad,
            loose=loose,
            cx=float(cx),
            cy=float(cy),
            ux=float(unit[0]),
            uy=float(unit[1]),
            height=min(length, height) if loose else height,
        )

    def toward(self, unit: np.ndarray) -> np.ndarray:
        """줄 방향 `unit`에 맞춘 사각형. 방향이 정해진 박스는 축은 그대로 두고 앞뒤만 맞춘다."""
        if self.loose:
            return orient(self.quad, self.box.text, prefer=unit)[0]
        if self.ux * unit[0] + self.uy * unit[1] < 0:
            return np.roll(self.quad, 2, axis=0)
        return self.quad


_COS_TOLERANCE = math.cos(math.radians(ANGLE_TOLERANCE_DEG))


def _same_line(a: _Item, b: _Item) -> bool:
    if a.loose and b.loose:
        ux, uy = 1.0, 0.0
    elif a.loose:
        ux, uy = b.ux, b.uy
    elif b.loose:
        ux, uy = a.ux, a.uy
    else:
        cos = a.ux * b.ux + a.uy * b.uy
        if abs(cos) < _COS_TOLERANCE:
            return False
        sign = 1.0 if cos >= 0 else -1.0
        ux, uy = a.ux + sign * b.ux, a.uy + sign * b.uy
    if min(a.height, b.height) < MIN_HEIGHT_RATIO * max(a.height, b.height):
        return False
    # 두 중심을 잇는 벡터를 글자 방향의 수직 축에 투영한 길이
    offset = abs((b.cy - a.cy) * ux - (b.cx - a.cx) * uy) / math.hypot(ux, uy)
    return offset < OFFSET_RATIO * (a.height + b.height) / 2


def _group(items: list[_Item]) -> list[list[_Item]]:
    parent = list(range(len(items)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if find(i) != find(j) and _same_line(items[i], items[j]):
                parent[find(j)] = find(i)
    groups: dict[int, list[_Item]] = {}
    for i, item in enumerate(items):
        groups.setdefault(find(i), []).append(item)
    return list(groups.values())


@dataclass
class _Line:
    boxes: list[TextBox]
    direction: np.ndarray
    center: np.ndarray
    length: float


def _make_line(group: list[_Item]) -> _Line:
    # 줄 방향: 방향을 아는 박스들의 길이 가중 평균 (하나도 없으면 전부로)
    sure = [item for item in group if not item.loose] or group
    reference = geometry.direction(sure[0].quad)
    total = np.zeros(2)
    for item in sure:
        unit = geometry.direction(item.quad)
        total += geometry.side_lengths(item.quad)[0] * (unit if unit @ reference >= 0 else -unit)
    line_dir = total / max(float(np.linalg.norm(total)), 1e-9)
    if not _reads_forward(line_dir):
        line_dir = -line_dir

    quads = sorted(
        ((item.toward(line_dir), item) for item in group),
        key=lambda pair: float(pair[0].mean(axis=0) @ line_dir),
    )
    boxes = [TextBox(text=item.box.text, confidence=item.box.confidence, quad=q) for q, item in quads]
    return _Line(
        boxes=boxes,
        direction=line_dir,
        center=np.mean([box.quad.mean(axis=0) for box in boxes], axis=0),
        length=sum(geometry.side_lengths(box.quad)[0] for box in boxes),
    )
