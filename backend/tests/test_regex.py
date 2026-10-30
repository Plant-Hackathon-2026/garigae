import pytest

from backend.detectors.base import PiiType, TextLine
from backend.detectors.regex import RegexDetector

detector = RegexDetector()


def found(text: str) -> list[tuple[str, PiiType]]:
    return [(text[s.start : s.end], s.type) for s in detector.detect([TextLine(id=0, text=text)])]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("연락처: 010-1234-5678", "010-1234-5678"),
        ("내 번호 바뀌었어 010 2345 6789", "010 2345 6789"),
        ("01012345678", "01012345678"),
        ("011-123-4567", "011-123-4567"),
    ],
)
def test_finds_mobile_phone(text, expected):
    assert found(text) == [(expected, PiiType.PHONE)]


@pytest.mark.parametrize(
    "text",
    [
        "주문번호 2026-1030-0001",
        "운송장번호 1234-5678-9012",
        "9010-1234-56789",  # 앞뒤가 숫자로 이어짐
    ],
)
def test_ignores_non_phone(text):
    assert found(text) == []
