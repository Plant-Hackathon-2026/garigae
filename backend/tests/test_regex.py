"""정규식 감지기 테스트. 항목별로 맞아야 하는 예 / 틀려야 하는 예 표를 둔다.

번호는 모두 가짜다.
"""

import pytest

from backend.detectors.base import PiiType, TextLine
from backend.detectors.regex import (
    SCORE_VALIDATED,
    RegexDetector,
)

detector = RegexDetector()


def make_lines(texts: list[str]) -> list[TextLine]:
    return [TextLine(id=i, text=text) for i, text in enumerate(texts)]


def found(text: str) -> list[tuple[str, PiiType]]:
    return [(text[s.start : s.end], s.type) for s in detector.detect([TextLine(id=0, text=text)])]


def score_of(text: str) -> float:
    [span] = detector.detect([TextLine(id=0, text=text)])
    return span.score


# --- 주민등록번호·외국인등록번호 ---------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected", "pii_type"),
    [
        ("900101-1234567", "900101-1234567", PiiType.RESIDENT_ID),
        ("주민등록번호 900101-2234567", "900101-2234567", PiiType.RESIDENT_ID),
        ("900101 1234567", "900101 1234567", PiiType.RESIDENT_ID),  # 공백 구분
        ("9001011234567", "9001011234567", PiiType.RESIDENT_ID),  # 구분자 없음
        ("900101 -1234567", "900101 -1234567", PiiType.RESIDENT_ID),  # 박스 사이 공백
        ("050505-3123456", "050505-3123456", PiiType.RESIDENT_ID),  # 2000년대생
        ("000229-3123456", "000229-3123456", PiiType.RESIDENT_ID),  # 2000년은 윤년
        ("960229-1123456", "960229-1123456", PiiType.RESIDENT_ID),  # 1996년은 윤년
        ("991231-4123456", "991231-4123456", PiiType.RESIDENT_ID),
        ("900101-5123456", "900101-5123456", PiiType.FOREIGNER_ID),
        ("외국인등록번호 851120-6123456", "851120-6123456", PiiType.FOREIGNER_ID),
        ("010101-7123456", "010101-7123456", PiiType.FOREIGNER_ID),
        ("040229-8123456", "040229-8123456", PiiType.FOREIGNER_ID),
    ],
)
def test_finds_resident_id(text, expected, pii_type):
    assert found(text) == [(expected, pii_type)]
    assert score_of(text) == SCORE_VALIDATED


@pytest.mark.parametrize(
    "text",
    [
        "901301-1234567",  # 13월
        "900001-1234567",  # 0월
        "900100-1234567",  # 0일
        "900230-1234567",  # 2월 30일
        "900431-1234567",  # 4월 31일
        "000229-1234567",  # 1900년은 윤년이 아님
        "010229-3123456",  # 2001년은 윤년이 아님
        "900101-9234567",  # 뒷자리 첫 숫자 9
        "900101-0234567",  # 뒷자리 첫 숫자 0
        "1900101-1234567",  # 앞이 숫자로 이어짐
        "900101-12345678",  # 뒤가 숫자로 이어짐
        "900101-123456",  # 뒷자리가 짧음
        "900101--1234567",  # 구분자가 두 개
    ],
)
def test_ignores_non_resident_id(text):
    assert found(text) == []


# --- 전화번호 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("연락처: 010-1234-5678", "010-1234-5678"),
        ("내 번호 바뀌었어 010 2345 6789", "010 2345 6789"),
        ("01012345678", "01012345678"),
        ("011-123-4567", "011-123-4567"),
        ("010.1234.5678", "010.1234.5678"),
        ("010 -1234 -5678", "010 -1234 -5678"),  # 박스 사이 공백
        ("전화 010-1234-5678입니다", "010-1234-5678"),
    ],
)
def test_finds_phone(text, expected):
    assert found(text) == [(expected, PiiType.PHONE)]
    assert score_of(text) == SCORE_VALIDATED


@pytest.mark.parametrize(
    "text",
    [
        "주문번호 2026-1030-0001",
        "운송장번호 1234-5678-9012",
        "9010-1234-56789",  # 앞뒤가 숫자로 이어짐
        "010-1234-56789",  # 뒤가 숫자로 이어짐
        "010-1234-5678-12",  # 더 긴 번호의 일부
        "010-12-5678",
    ],
)
def test_ignores_non_phone(text):
    assert found(text) == []
