"""파이프라인 테스트. OCR은 가짜 엔진으로 바꿔 끼워 torch 없이 돌린다."""

import math

import cv2
import numpy as np
import pytest
from fakes import FakeOcrEngine, along, make_box, signed_area, tilted_box

from backend.detectors.base import PiiSpan, PiiType
from backend.detectors.registry import build_detectors
from backend.pipeline import Pipeline, merge_spans

IMAGE = np.zeros((400, 800, 3), np.uint8)


def run(boxes, detectors=None, types=None):
    pipeline = Pipeline(FakeOcrEngine(boxes), detectors or build_detectors(["regex"]))
    return pipeline.run(IMAGE, types=types)


def covers(polygon, points) -> bool:
    contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    return all(cv2.pointPolygonTest(contour, (float(x), float(y)), True) >= -1e-3 for x, y in points)


class FixedDetector:
    def __init__(self, name, spans):
        self.name = name
        self.spans = spans

    def detect(self, lines):
        return [PiiSpan(source=self.name, **span) for span in self.spans]


def test_phone_inside_labelled_box():
    result = run([make_box("연락처: 010-1234-5678", 40, 100, 400, 40, confidence=0.8)])
    [detection] = result.detections
    assert detection.id == "d1"
    assert (detection.type, detection.label, detection.text, detection.source) == (
        PiiType.PHONE,
        "전화번호",
        "010-1234-5678",
        "regex",
    )
    assert detection.confidence == pytest.approx(0.8 * 0.9)
    xs = [x for x, _ in detection.polygon]
    # 라벨(`연락처:`)은 가리지 않고 번호는 박스 끝까지
    assert min(xs) > 40 + 400 * 0.2
    assert max(xs) >= 440
    assert signed_area(detection.polygon) > 0


def test_phone_split_into_boxes_is_one_detection():
    boxes = [make_box(t, 100 + 100 * i, 100, 80, 40) for i, t in enumerate(["010", "1234", "5678"])]
    [detection] = run(boxes).detections
    assert detection.text == "010 1234 5678"
    assert covers(detection.polygon, [p for box in boxes for p in box.points])


def test_tilted_split_phone():
    angle = 20
    boxes = [
        tilted_box(text, along((150, 300), angle, 100 * i), 80, 40, angle)
        for i, text in enumerate(["010", "2345", "6789"])
    ]
    [detection] = run(list(reversed(boxes))).detections
    assert detection.text == "010 2345 6789"
    assert covers(detection.polygon, [p for box in boxes for p in box.points])
    (x0, y0), (x1, y1) = detection.polygon[0], detection.polygon[1]
    assert math.degrees(math.atan2(y0 - y1, x1 - x0)) == pytest.approx(angle, abs=1)


def test_confidence_uses_lowest_box():
    boxes = [
        make_box(t, 100 + 100 * i, 100, 80, 40, confidence=c)
        for i, (t, c) in enumerate([("010", 0.9), ("1234", 0.5), ("5678", 0.8)])
    ]
    [detection] = run(boxes).detections
    assert detection.confidence == pytest.approx(0.5 * 0.9)


def test_types_filter():
    boxes = [make_box("연락처: 010-1234-5678", 40, 100, 400, 40)]
    assert run(boxes, types={PiiType.EMAIL}).detections == []
    assert len(run(boxes, types={PiiType.PHONE}).detections) == 1
    assert len(run(boxes, types=None).detections) == 1


def test_overlapping_spans_from_detectors_are_merged():
    boxes = [make_box("받는 분 김가상 010-1234-5678", 40, 100, 500, 40, confidence=1.0)]
    detectors = [
        FixedDetector("a", [{"line_id": 0, "start": 5, "end": 9, "type": PiiType.NAME, "score": 0.6}]),
        FixedDetector("b", [{"line_id": 0, "start": 8, "end": 22, "type": PiiType.PHONE, "score": 0.9}]),
    ]
    [detection] = run(boxes, detectors).detections
    assert detection.text == "김가상 010-1234-5678"
    assert (detection.type, detection.source) == (PiiType.PHONE, "b")
    assert detection.confidence == pytest.approx(0.9)


def test_types_filter_applies_before_merging():
    boxes = [make_box("받는 분 김가상 010-1234-5678", 40, 100, 500, 40)]
    detectors = [
        FixedDetector("a", [{"line_id": 0, "start": 5, "end": 9, "type": PiiType.NAME, "score": 0.6}]),
        FixedDetector("b", [{"line_id": 0, "start": 8, "end": 22, "type": PiiType.PHONE, "score": 0.9}]),
    ]
    [detection] = run(boxes, detectors, types={PiiType.NAME}).detections
    assert (detection.text, detection.type) == ("김가상 ", PiiType.NAME)


def test_spans_outside_lines_are_dropped():
    boxes = [make_box("010-1234-5678", 40, 100, 300, 40)]
    detectors = [
        FixedDetector("bad", [{"line_id": 3, "start": 0, "end": 2, "type": PiiType.NAME, "score": 1}]),
        FixedDetector("long", [{"line_id": 0, "start": 0, "end": 99, "type": PiiType.NAME, "score": 1}]),
    ]
    assert run(boxes, detectors).detections == []


def test_text_regions_follow_reading_order():
    boxes = [
        make_box("둘째 줄", 40, 200, 160, 40, confidence=0.7),
        make_box("첫째", 40, 100, 90, 40, confidence=0.9),
        make_box("줄", 150, 100, 50, 40, confidence=0.8),
    ]
    regions = run(boxes).text_regions
    assert [(r.id, r.text, r.confidence) for r in regions] == [
        ("t1", "첫째", 0.9),
        ("t2", "줄", 0.8),
        ("t3", "둘째 줄", 0.7),
    ]
    assert regions[0].polygon == [(40.0, 100.0), (130.0, 100.0), (130.0, 140.0), (40.0, 140.0)]


def test_polygons_stay_inside_image():
    result = run([make_box("010-1234-5678", 560, 370, 260, 40)])
    for item in [*result.detections, *result.text_regions]:
        for x, y in item.polygon:
            assert 0 <= x <= 800 and 0 <= y <= 400


def test_no_text():
    result = run([])
    assert result.detections == [] and result.text_regions == []


def span(start, end, score=0.5, pii_type=PiiType.NAME, line_id=0, source="a"):
    return PiiSpan(line_id=line_id, start=start, end=end, type=pii_type, score=score, source=source)


def test_merge_spans_widens_overlaps():
    merged = merge_spans([span(0, 5), span(3, 8), span(7, 10), span(12, 14)])
    assert [(s.start, s.end) for s in merged] == [(0, 10), (12, 14)]


def test_merge_spans_keeps_touching_and_other_lines_apart():
    merged = merge_spans([span(0, 5), span(5, 8), span(2, 6, line_id=1)])
    assert [(s.line_id, s.start, s.end) for s in merged] == [(0, 0, 5), (0, 5, 8), (1, 2, 6)]


def test_merge_spans_takes_highest_score():
    merged = merge_spans(
        [
            span(0, 5, 0.5, PiiType.NAME, source="rules"),
            span(2, 9, 0.9, PiiType.PHONE, source="regex"),
            span(4, 6, 0.9, PiiType.ACCOUNT, source="ner"),
        ]
    )
    [result] = merged
    assert (result.start, result.end, result.type, result.source, result.score) == (
        0,
        9,
        PiiType.PHONE,
        "regex",
        0.9,
    )
