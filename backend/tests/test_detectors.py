"""모든 감지기가 지켜야 하는 약속 (감지기 공통 테스트)."""

import sys

import pytest

from backend.detectors.registry import build_detectors

ALL_DETECTORS = ["regex", "rules", "ner"]


@pytest.mark.parametrize("name", ALL_DETECTORS)
def test_spans_point_inside_lines(name, sample_lines):
    [detector] = build_detectors([name])
    assert detector.name == name
    for lines in sample_lines.values():
        texts = {line.id: line.text for line in lines}
        for span in detector.detect(lines):
            assert span.line_id in texts
            assert 0 <= span.start < span.end <= len(texts[span.line_id])
            assert span.source == name


@pytest.mark.parametrize("name", ALL_DETECTORS)
def test_empty_input(name):
    [detector] = build_detectors([name])
    assert detector.detect([]) == []


def test_registry_keeps_order_and_drops_duplicates():
    detectors = build_detectors(["rules", "regex", "rules"])
    assert [d.name for d in detectors] == ["rules", "regex"]


def test_ner_is_not_imported_when_disabled(monkeypatch):
    # NER 의존성이 없어도 ner를 끄면 서버가 떠야 한다
    monkeypatch.delitem(sys.modules, "backend.detectors.ner", raising=False)
    build_detectors(["regex", "rules"])
    assert "backend.detectors.ner" not in sys.modules
