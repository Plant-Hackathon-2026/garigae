import math

import cv2
import numpy as np
import pytest
from fakes import rotated_rect, signed_area

from backend import geometry


def inside(polygon, point, tolerance=1e-6) -> bool:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), True) >= -tolerance


def angle_of(quad) -> float:
    unit = geometry.direction(np.asarray(quad))
    return math.degrees(math.atan2(-unit[1], unit[0]))


@pytest.mark.parametrize(
    ("char", "weight"),
    [("가", 1.0), ("漢", 1.0), ("0", 0.65), ("a", 0.65), ("Z", 0.65), (" ", 0.35), ("-", 0.4), (":", 0.4), ("@", 0.4)],
)
def test_char_weight(char, weight):
    assert geometry.char_weight(char) == weight


def box(x, y, width, height):
    return rotated_rect(x + width / 2, y + height / 2, width, height, 0)


def test_whole_box_gets_margins():
    quad = box(100, 100, 300, 40)
    result = geometry.span_quad(quad, "01012345678", 0, 11)
    xs, ys = result[:, 0], result[:, 1]
    # 박스를 다 덮고, 위아래로 높이의 10%
    assert xs.min() < 100 and xs.max() > 400
    assert ys.min() == pytest.approx(96) and ys.max() == pytest.approx(144)


def test_partial_span_uses_char_widths():
    # 한글 4자 + 숫자 4자: 숫자는 폭 비율로 뒤쪽 약 40%
    quad = box(0, 0, 400, 40)
    text = "가나다라0000"
    result = geometry.span_quad(quad, text, 4, 8)
    start = result[:, 0].min()
    total = geometry.text_width(text)
    # 셋째 한글 글자까지는 가리지 않고, 숫자 시작 위치보다는 앞에서 시작 (여백)
    assert 3 / total * 400 < start < 4 / total * 400
    assert result[:, 0].max() >= 400


def test_partial_span_in_tilted_box_keeps_angle():
    quad = rotated_rect(300, 300, 400, 40, 20)
    result = geometry.span_quad(quad, "연락처: 010-1234-5678", 5, 18)
    assert angle_of(result) == pytest.approx(20)
    # 위·아래 변이 원래 박스 변과 나란하다
    for a, b in ((result[0], result[1]), (result[3], result[2])):
        edge = (b - a) / np.linalg.norm(b - a)
        assert edge == pytest.approx(geometry.direction(quad))
    # 박스 뒤쪽 절반 근처
    center = result.mean(axis=0)
    assert (center - quad.mean(axis=0)) @ geometry.direction(quad) > 0


def test_span_over_several_boxes_is_one_rotated_rectangle():
    angle = 20
    unit = np.array([math.cos(math.radians(angle)), -math.sin(math.radians(angle))])
    quads, texts = [], ["010", "1234", "5678"]
    for i, text in enumerate(texts):
        center = np.array([200.0, 300.0]) + unit * (i * 110)
        quads.append(rotated_rect(center[0], center[1], 90, 40, angle))
    pieces = [(q, t, 0, len(t)) for q, t in zip(quads, texts)]
    polygon = geometry.span_polygon(pieces, (1000, 1000))
    assert polygon.shape == (4, 2)
    assert signed_area(polygon) > 0  # 시계 방향
    assert angle_of(polygon) == pytest.approx(angle, abs=1)
    for quad in quads:
        for point in quad:
            assert inside(polygon, point)
    # 첫 꼭짓점이 글자 기준 왼쪽 위
    expected_first = min(polygon, key=lambda p: p @ unit + p @ np.array([-unit[1], unit[0]]))
    assert polygon[0] == pytest.approx(expected_first)


@pytest.mark.parametrize("angle", [0, 20, 60, -30])
def test_order_by_direction_starts_at_text_top_left(angle):
    rect = rotated_rect(200, 200, 200, 40, angle)
    shuffled = rect[[2, 0, 3, 1]]
    unit = geometry.direction(rect)
    assert geometry.order_by_direction(shuffled, unit) == pytest.approx(rect)


def test_polygon_is_clipped_to_image():
    quad = box(-20, -10, 200, 40)
    polygon = geometry.span_polygon([(quad, "0101234", 0, 7)], (150, 100))
    assert polygon[:, 0].min() == 0 and polygon[:, 0].max() == 150
    assert polygon[:, 1].min() == 0
    assert polygon.shape == (4, 2)


def test_to_points_rounds():
    assert geometry.to_points(np.array([[1.234, 2.0], [3, 4], [5, 6], [7.06, 8]])) == [
        (1.2, 2.0),
        (3.0, 4.0),
        (5.0, 6.0),
        (7.1, 8.0),
    ]
