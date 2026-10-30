import math
import random

import numpy as np
import pytest
from fakes import along, make_box, tilted_box

from backend import geometry
from backend.lines import build_lines, orient


def texts(boxes) -> list[str]:
    return [layout.line.text for layout in build_lines(boxes)]


def angle_of(quad) -> float:
    """글자 방향 각도 (화면에서 반시계가 양수)."""
    unit = geometry.direction(quad)
    return math.degrees(math.atan2(-unit[1], unit[0]))


def words_along(words, start, angle, height=40, gap=18):
    """`start`에서 `angle`° 방향으로 단어 박스를 이어 놓는다. 단어 길이는 글자 폭 가중치에 비례."""
    boxes, distance = [], 0.0
    for word in words:
        width = geometry.text_width(word) * height * 0.8 + 10
        center = along(start, angle, distance + width / 2)
        boxes.append(tilted_box(word, center, width, height, angle))
        distance += width + gap
    return boxes


def test_empty():
    assert build_lines([]) == []


def test_joins_boxes_of_a_line_in_reading_order():
    boxes = [
        make_box("5678", 400, 100, 80, 40),
        make_box("연락처:", 40, 102, 110, 40),
        make_box("1234", 300, 99, 80, 40),
        make_box("010", 200, 101, 70, 40),
    ]
    [layout] = build_lines(boxes)
    assert layout.line.id == 0
    assert layout.line.text == "연락처: 010 1234 5678"
    assert [box.text for box in layout.boxes] == ["연락처:", "010", "1234", "5678"]


def test_lines_are_ordered_top_to_bottom():
    boxes = [
        make_box("셋째 줄", 40, 220, 160, 40),
        make_box("첫째 줄", 40, 20, 160, 40),
        make_box("둘째", 40, 120, 90, 40),
        make_box("줄", 150, 121, 50, 40),
    ]
    layouts = build_lines(boxes)
    assert [(layout.line.id, layout.line.text) for layout in layouts] == [
        (0, "첫째 줄"),
        (1, "둘째 줄"),
        (2, "셋째 줄"),
    ]


@pytest.mark.parametrize("angle", [8, 20, -15])
def test_tilted_line_is_one_line(angle):
    # EasyOCR는 약 5.7° 넘게 기울어진 박스를 합치지 않고 단어마다 따로 준다
    boxes = words_along(["내", "번호", "010", "2345", "6789"], (60, 300), angle)
    ys = [np.mean([p[1] for p in box.points]) for box in boxes]
    # 세로 위치만으로 묶으면 한 줄이 안 되는 기울기다
    assert max(ys) - min(ys) > 40
    random.Random(0).shuffle(boxes)
    assert texts(boxes) == ["내 번호 010 2345 6789"]


def test_parallel_tilted_lines_stay_apart():
    angle = 20
    first = words_along(["연락처", "안내"], (60, 200), angle)
    # 글자 방향에 수직으로 높이의 1.5배 아래
    down = (math.sin(math.radians(angle)), math.cos(math.radians(angle)))
    second = words_along(["휴대폰:", "010-2468-1357"], (60 + 60 * down[0], 200 + 60 * down[1]), angle)
    assert texts(second + first) == ["연락처 안내", "휴대폰: 010-2468-1357"]


def test_steep_box_reads_along_long_side():
    # 60°: EasyOCR의 첫 꼭짓점(x+y 최소)이 글자 기준 왼쪽 위가 아니다
    box = tilted_box("010-1234-5678", (300, 300), 300, 40, 60)
    quad, ambiguous = orient(np.asarray(box.points), box.text)
    assert not ambiguous
    assert angle_of(quad) == pytest.approx(60, abs=0.5)
    assert not np.allclose(quad[0], box.points[0])
    # 글자 기준 왼쪽 위·오른쪽 위·오른쪽 아래·왼쪽 아래
    assert np.allclose(quad, _expected_quad((300, 300), 300, 40, 60))


def test_steep_line_keeps_reading_order():
    boxes = words_along(["연락처:", "010-1234-5678"], (100, 500), 60)
    [layout] = build_lines(list(reversed(boxes)))
    assert layout.line.text == "연락처: 010-1234-5678"
    for box in layout.boxes:
        assert angle_of(box.quad) == pytest.approx(60, abs=0.5)


def test_narrow_short_box_is_not_turned_vertical():
    # `1` 한 글자 박스는 세로가 더 길지만 글자 방향은 가로다
    box = make_box("1", 200, 100, 22, 40)
    quad, _ = orient(np.asarray(box.points), box.text)
    assert angle_of(quad) == pytest.approx(0)
    assert texts([make_box("번호", 100, 100, 70, 40), box]) == ["번호 1"]


def test_tall_long_box_reads_top_to_bottom():
    # 같은 모양이라도 텍스트가 길면 세로로 쓴 글자로 본다 (90° 돌아간 이미지)
    box = make_box("010-1234-5678", 200, 100, 40, 300)
    quad, _ = orient(np.asarray(box.points), box.text)
    assert geometry.direction(quad) == pytest.approx([0, 1])


def test_square_box_follows_line_direction():
    angle = 60
    boxes = words_along(["연락처", "가", "010-1234-5678"], (100, 600), angle)
    [layout] = build_lines(boxes)
    assert layout.line.text == "연락처 가 010-1234-5678"
    assert angle_of(layout.boxes[1].quad) == pytest.approx(angle, abs=0.5)


def test_axis_aligned_square_box_joins_tilted_line():
    # EasyOCR(CRAFT)는 정사각형에 가까운 단어를 기울어져 있어도 축에 나란한 외접 사각형으로 준다
    angle = 20
    boxes = words_along(["문의", "시간", "평일", "09:00-18:00"], (100, 400), angle)
    for i in (0, 1):
        xs = [p[0] for p in boxes[i].points]
        ys = [p[1] for p in boxes[i].points]
        boxes[i] = make_box(boxes[i].text, min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
    assert texts(boxes) == ["문의 시간 평일 09:00-18:00"]


def test_char_map_points_into_boxes():
    boxes = [make_box("010", 100, 100, 60, 40), make_box("1234", 180, 100, 80, 40)]
    [layout] = build_lines(boxes)
    assert layout.line.text == "010 1234"
    assert layout.char_map == ((0, 0), (0, 1), (0, 2), None, (1, 0), (1, 1), (1, 2), (1, 3))


def test_pieces_inside_one_box():
    [layout] = build_lines([make_box("연락처: 010-1234-5678", 40, 100, 400, 40)])
    start = layout.line.text.index("010")
    [(quad, text, s, e)] = layout.pieces(start, start + 13)
    assert (text[s:e], s, e) == ("010-1234-5678", 5, 18)
    assert quad is layout.boxes[0].quad


def test_pieces_across_boxes():
    boxes = [make_box(t, 100 + 90 * i, 100, 80, 40) for i, t in enumerate(["010", "1234", "5678"])]
    [layout] = build_lines(boxes)
    pieces = layout.pieces(0, len(layout.line.text))
    assert [(text[s:e]) for _, text, s, e in pieces] == ["010", "1234", "5678"]
    # 구간이 박스 일부만 걸치면 그 부분만
    assert [(text[s:e]) for _, text, s, e in layout.pieces(2, 6)] == ["0", "12"]
    # 박스 사이 공백만 가리키면 조각이 없다
    assert layout.pieces(3, 4) == []
    assert [box.text for box in layout.boxes_in(2, 6)] == ["010", "1234"]


def test_skips_boxes_without_text():
    boxes = [make_box("", 40, 100, 50, 40), make_box("  ", 100, 100, 50, 40), make_box("가상", 160, 100, 70, 40)]
    assert texts(boxes) == ["가상"]


def _expected_quad(center, width, height, angle):
    from fakes import rotated_rect

    return rotated_rect(center[0], center[1], width, height, angle)
