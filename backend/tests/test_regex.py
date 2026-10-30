"""정규식 감지기 테스트. 항목별로 맞아야 하는 예 / 틀려야 하는 예 표를 둔다.

번호는 모두 가짜다. 카드번호는 결제 테스트용으로 공개된 번호(4111 1111 1111 1111 등)를 쓴다.
"""

import pytest

from backend.detectors.base import PiiType, TextLine
from backend.detectors.regex import (
    SCORE_MASKED_CARD,
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
        ("02-123-4567", "02-123-4567"),  # 서울
        ("02-1234-5678", "02-1234-5678"),
        ("(02) 1234-5678", "(02) 1234-5678"),
        ("TEL 02)1234-5678", "02)1234-5678"),
        ("031-123-4567", "031-123-4567"),
        ("044-123-4567", "044-123-4567"),  # 세종
        ("064-1234-5678", "064-1234-5678"),
        ("070-1234-5678", "070-1234-5678"),  # 인터넷 전화
        ("0505-123-4567", "0505-123-4567"),  # 안심번호
        ("0504-1234-5678", "0504-1234-5678"),
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
        "고객센터 1588-1234",  # 대표번호
        "1644-1234",
        "1899-1234",
        "080-123-4567",  # 수신자 부담 (업체 번호)
        "034-123-4567",  # 없는 지역번호
        "02-12-3456",  # 자릿수 부족
        "010-12-5678",
    ],
)
def test_ignores_non_phone(text):
    assert found(text) == []


# --- 이메일 -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("문의: help@example.com", "help@example.com"),
        ("hong.test@example.co.kr", "hong.test@example.co.kr"),
        ("user_01+tag@mail-server.example.com", "user_01+tag@mail-server.example.com"),
        ("메일help@example.com으로 보내 주세요", "help@example.com"),
        ("hong.test @example.com", "hong.test @example.com"),  # 박스 사이 공백
        ("01012345678@example.com", "01012345678@example.com"),  # 아이디가 전화번호여도 이메일 하나
    ],
)
def test_finds_email(text, expected):
    assert found(text) == [(expected, PiiType.EMAIL)]


@pytest.mark.parametrize(
    "text",
    [
        "help@example",  # 최상위 도메인 없음
        "@example.com",
        "help@.com",
        # 실제 EasyOCR가 hong.test@example.com을 이렇게 읽은 적이 있다. 빠진 `.`을 추측해 채우지 않는다
        "hong test@examplecom",
    ],
)
def test_ignores_non_email(text):
    assert found(text) == []


# --- 카드번호 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("4111-1111-1111-1111", "4111-1111-1111-1111"),
        ("카드 4242 4242 4242 4242 승인", "4242 4242 4242 4242"),
        ("5555555555554444", "5555555555554444"),  # 구분자 없음
        ("1234-5678-9012-3452", "1234-5678-9012-3452"),
        ("3782-822463-10005", "3782-822463-10005"),  # 아멕스
        ("3714 496353 98431", "3714 496353 98431"),
    ],
)
def test_finds_card(text, expected):
    assert found(text) == [(expected, PiiType.CARD)]
    assert score_of(text) == SCORE_VALIDATED


@pytest.mark.parametrize(
    "text",
    [
        "4111-1111-1111-1112",  # Luhn 실패
        "3782-822463-10006",  # Luhn 실패
        "4111-111111-11116",  # Luhn은 맞지만 4-6-5인데 아멕스(34·37)가 아님
        "94111-1111-1111-1111",  # 앞이 숫자로 이어짐
        "4111-1111-1111-11111",  # 뒤가 숫자로 이어짐
        "4111-1111-1111-1111-2222",  # 더 긴 번호의 일부
    ],
)
def test_ignores_non_card(text):
    assert found(text) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("카드번호 1234-56**-****-7890", "1234-56**-****-7890"),
        ("1234-****-****-7890", "1234-****-****-7890"),
        ("**** **** **** 7890", "**** **** **** 7890"),
        ("4111-11XX-XXXX-1111", "4111-11XX-XXXX-1111"),  # X로 가림
        ("123456******7890", "123456******7890"),
        ("3782-82****-*0005", "3782-82****-*0005"),  # 아멕스
    ],
)
def test_finds_masked_card(text, expected):
    assert found(text) == [(expected, PiiType.CARD)]
    assert score_of(text) == SCORE_MASKED_CARD


@pytest.mark.parametrize(
    "text",
    [
        "****-****-****-****",  # 숫자가 하나도 없음
        "1234-56**-****-78901",  # 뒤가 숫자로 이어짐
        "1234-56**-****",  # 그룹 부족
    ],
)
def test_ignores_non_masked_card(text):
    assert found(text) == []
