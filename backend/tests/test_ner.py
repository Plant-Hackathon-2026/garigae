"""NER 감지기 테스트. NER 담당자가 구현하면서 채운다.

`test_finds_expected`는 구현 전이라 xfail로 표시해 뒀다. 구현 후 표시를 지운다.
"""

import pytest

from backend.detectors.base import PiiType
from backend.detectors.ner import NerDetector


@pytest.fixture(scope="module")
def detector():
    # 모델 로딩이 무거우면 모듈당 한 번만 만든다
    return NerDetector()


# (샘플 문서, 감지 구간에 들어 있어야 하는 문자열, 종류)
EXPECTED = [
    ("id_card", "홍길동", PiiType.NAME),
    ("delivery_label", "김철수", PiiType.NAME),
    ("delivery_label", "이영희", PiiType.NAME),
    ("messenger", "홍길동", PiiType.NAME),
    ("receipt", "박민수", PiiType.NAME),
    ("bank_notice", "최지우", PiiType.NAME),
    ("id_card", "서울특별시 가상구 테스트로 12", PiiType.ADDRESS),
    ("delivery_label", "부산광역시 해운대구 가상로 34, 101동 202호", PiiType.ADDRESS),
]


@pytest.mark.xfail(reason="NER 구현 전")
@pytest.mark.parametrize(("document", "expected", "pii_type"), EXPECTED)
def test_finds_expected(detector, sample_lines, document, expected, pii_type):
    lines = sample_lines[document]
    found = [
        lines[span.line_id].text[span.start : span.end]
        for span in detector.detect(lines)
        if span.type == pii_type
    ]
    assert any(expected in text for text in found), found


def test_returns_spans(detector, sample_lines):
    for lines in sample_lines.values():
        assert isinstance(detector.detect(lines), list)
