"""정규식 감지기 테스트. 항목별로 맞아야 하는 예 / 틀려야 하는 예 표를 둔다.

번호는 모두 가짜다. 카드번호는 결제 테스트용으로 공개된 번호(4111 1111 1111 1111 등)를 쓴다.
"""

import itertools

import pytest

from backend.detectors.base import PiiType, TextLine
from backend.detectors.regex import (
    OCR_FIX_PENALTY,
    SCORE_ACCOUNT,
    SCORE_MASKED_CARD,
    SCORE_PASSPORT,
    SCORE_PASSPORT_WITH_KEYWORD,
    SCORE_VALIDATED,
    RegexDetector,
    normalize_ocr,
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


# --- 여권번호 ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("M12345678", "M12345678"),  # 구형
        ("S12345678", "S12345678"),
        ("O12345678", "O12345678"),  # 관용 여권
        ("M123A4567", "M123A4567"),  # 차세대
        ("PM M123B4567 KOR", "M123B4567"),
        ("여권번호 G12345678", "G12345678"),
    ],
)
def test_finds_passport(text, expected):
    assert found(text) == [(expected, PiiType.PASSPORT)]


@pytest.mark.parametrize(
    "text",
    [
        "A12345678",  # 쓰지 않는 영문
        "M1234567",  # 7자리
        "M123456789",  # 9자리
        "XM12345678",  # 앞에 영문이 붙음
        "M12345678X",  # 뒤에 영문이 붙음
        "m12345678",  # 소문자
        "M123AB4567",  # 차세대 형식에서 영문이 두 개
    ],
)
def test_ignores_non_passport(text):
    assert found(text) == []


def test_passport_keyword_raises_score():
    assert score_of("M12345678") == SCORE_PASSPORT
    assert score_of("여권번호 M12345678") == SCORE_PASSPORT_WITH_KEYWORD
    assert score_of("Passport No. M12345678") == SCORE_PASSPORT_WITH_KEYWORD
    assert score_of("PASSPORT M12345678") == SCORE_PASSPORT_WITH_KEYWORD


@pytest.mark.parametrize(
    ("texts", "expected_score"),
    [
        (["여권 PASSPORT", "종류/Type", "M12345678"], SCORE_PASSPORT_WITH_KEYWORD),  # 2줄 위
        (["M12345678", "성명", "여권"], SCORE_PASSPORT_WITH_KEYWORD),  # 2줄 아래
        (["여권", "가", "나", "M12345678"], SCORE_PASSPORT),  # 3줄 위는 근처로 보지 않음
    ],
)
def test_passport_keyword_on_nearby_lines(texts, expected_score):
    [span] = detector.detect(make_lines(texts))
    assert span.type == PiiType.PASSPORT
    assert span.score == expected_score


# --- 운전면허번호 -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("11-12-345678-90", "11-12-345678-90"),  # 서울
        ("면허번호 26-99-000001-10", "26-99-000001-10"),  # 울산
        ("28-20-123456-01", "28-20-123456-01"),  # 경기북부
        ("13 15 123456 78", "13 15 123456 78"),  # 공백 구분
        ("서울 12-345678-90", "서울 12-345678-90"),  # 구형 지역명
        ("경기12-345678-90", "경기12-345678-90"),
        ("제주 95-123456-12", "제주 95-123456-12"),
    ],
)
def test_finds_driver_license(text, expected):
    assert found(text) == [(expected, PiiType.DRIVER_LICENSE)]
    assert score_of(text) == SCORE_VALIDATED


@pytest.mark.parametrize(
    "text",
    [
        "10-12-345678-90",  # 없는 지역 코드
        "27-12-345678-90",  # 결번
        "29-12-345678-90",
        "111234567890",  # 구분자 없는 12자리 (택배 송장번호와 겹침)
        "11-12-345678-901",  # 뒤가 숫자로 이어짐
        "가상 12-345678-90",  # 없는 지역명
    ],
)
def test_ignores_non_driver_license(text):
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


# --- 계좌번호 -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        (["신한은행 110-123-456789"], "110-123-456789"),
        (["국민은행 123456-78-901234 홍길동"], "123456-78-901234"),
        (["국민 123456-78-901234"], "123456-78-901234"),
        (["KB 123456-78-901234"], "123456-78-901234"),
        (["카카오뱅크 3333-01-1234567"], "3333-01-1234567"),
        (["새마을금고 9002-1234-5678-9"], "9002-1234-5678-9"),
        (["계좌: 1002-123-456789"], "1002-123-456789"),
        (["입금 302-1234-5678-91"], "302-1234-5678-91"),
        (["예금주 홍길동 123-456789-12345"], "123-456789-12345"),
        (["입금 계좌 안내", "110-123-456789"], "110-123-456789"),  # 바로 윗줄 키워드
        (["은행: 우리은행", "1002 - 123 - 456789"], "1002 - 123 - 456789"),
    ],
)
def test_finds_account(texts, expected):
    spans = detector.detect(make_lines(texts))
    assert [(texts[s.line_id][s.start : s.end], s.type) for s in spans] == [
        (expected, PiiType.ACCOUNT)
    ]
    assert [s.score for s in spans] == [SCORE_ACCOUNT]


@pytest.mark.parametrize(
    "texts",
    [
        ["주문번호 110-123-456789"],  # 키워드 없음
        ["운송장번호 1234-5678-9012"],
        ["사업자번호 123-45-67890"],
        ["계좌 안내", "기타 문의", "110-123-456789"],  # 키워드가 두 줄 위
        ["110-123-456789", "입금 계좌"],  # 키워드가 아랫줄
        ["계좌 123-456789"],  # 9자리
        ["계좌 1234-5678-9012-345"],  # 15자리
        ["계좌 110123456789"],  # 하이픈 그룹이 없음
        ["우리 집 1002-123-456789"],  # 은행 이름이 아닌 "우리"
    ],
)
def test_ignores_non_account(texts):
    assert detector.detect(make_lines(texts)) == []


def test_account_keyword_uses_line_id_not_list_position():
    above = [TextLine(id=10, text="입금 계좌"), TextLine(id=11, text="110-123-456789")]
    assert [s.type for s in detector.detect(above)] == [PiiType.ACCOUNT]
    # 목록에서는 바로 앞이지만 id가 이어지지 않으면 바로 윗줄이 아니다
    gap = [TextLine(id=10, text="입금 계좌"), TextLine(id=12, text="110-123-456789")]
    assert detector.detect(gap) == []


# --- OCR 오인식 보정 -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("O1O-l234-5678", "010-1234-5678"),
        ("010-1234-567O", "010-1234-5670"),
        ("o1o-I234-5678", "010-1234-5678"),
        ("010-1234-|234", "010-1234-1234"),
        ("010—1234—5678", "010-1234-5678"),
        ("010–1234–5678", "010-1234-5678"),
        ("010一1234一5678", "010-1234-5678"),
        ("010~1234~5678", "010-1234-5678"),
        ("O1O—l234—5678", "010-1234-5678"),  # 글자 치환 뒤에 숫자 사이가 된 대시
        ("연락처O1O-1234-5678", "연락처010-1234-5678"),  # 한글이 붙은 것은 숫자 문맥
    ],
)
def test_normalize_ocr_fixes_digit_context(text, expected):
    assert normalize_ocr(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "hong.test@example.com",
        "hello1",  # 영문 단어에 붙은 숫자
        "Iol",  # 진짜 숫자가 없음
        "Hello World 2026",
        "Tel 010",
        "M1234567O",  # 영문자가 붙은 구간
        "가나一다",  # 숫자 사이가 아닌 대시
        "10 ~ 20",  # 공백이 끼면 숫자 사이가 아님
        "| 이름 |",
    ],
)
def test_normalize_ocr_keeps_words(text):
    assert normalize_ocr(text) == text


@pytest.mark.parametrize(
    ("text", "expected", "pii_type"),
    [
        ("O1O-l234-5678", "O1O-l234-5678", PiiType.PHONE),
        ("연락처: 010一1234一5678", "010一1234一5678", PiiType.PHONE),
        ("9OO1O1-l234567", "9OO1O1-l234567", PiiType.RESIDENT_ID),
        ("4111 1111 1111 111l", "4111 1111 1111 111l", PiiType.CARD),
        ("4111—1111—1111—1111", "4111—1111—1111—1111", PiiType.CARD),
        ("카드 1234-56**-****-789O", "1234-56**-****-789O", PiiType.CARD),
        ("서울 l2-345678-9O", "서울 l2-345678-9O", PiiType.DRIVER_LICENSE),
        ("신한은행 11O-123-456789", "11O-123-456789", PiiType.ACCOUNT),
    ],
)
def test_finds_after_ocr_fix(text, expected, pii_type):
    # 보정한 사본에서 찾아도 문자 위치는 원본 기준이라, 잘라 낸 원본 글자에 오인식 글자가 그대로 있다
    assert found(text) == [(expected, pii_type)]


def test_ocr_fix_lowers_score_slightly():
    assert score_of("010-1234-5678") == SCORE_VALIDATED
    assert score_of("O1O-1234-5678") == pytest.approx(SCORE_VALIDATED - OCR_FIX_PENALTY)


def test_ocr_fix_does_not_break_original_match():
    # `|`를 1로 바꾸면 뒤에 숫자가 붙어 경계가 깨지지만, 원본에서 잡은 것이 남는다
    assert found("|010-1234-5678|") == [("010-1234-5678", PiiType.PHONE)]
    assert score_of("|010-1234-5678|") == SCORE_VALIDATED
    # 관용 여권의 O는 숫자로 바꾸지 않는다 (여권번호는 원본에서만 찾는다)
    assert found("O12345678") == [("O12345678", PiiType.PASSPORT)]


# --- 겹칠 때 우선순위 ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("texts", "expected"),
    [
        # 전화번호 모양이면 계좌 키워드가 있어도 전화번호
        (["계좌 010-1234-5678"], [("010-1234-5678", PiiType.PHONE)]),
        # 주민번호는 계좌 모양(6-7)이기도 하다
        (["예금주 900101-1234567"], [("900101-1234567", PiiType.RESIDENT_ID)]),
        # 운전면허번호도 계좌 모양이다
        (["계좌 11-12-345678-90"], [("11-12-345678-90", PiiType.DRIVER_LICENSE)]),
        (["입금", "서울 12-345678-90"], [("서울 12-345678-90", PiiType.DRIVER_LICENSE)]),
        # 이메일 아이디 자리의 전화번호를 따로 내보내지 않는다
        (["01012345678@example.com"], [("01012345678@example.com", PiiType.EMAIL)]),
        # 한 줄에 여러 개면 왼쪽부터
        (
            ["보내는 분: 이영희 010-9876-5432 / hong@example.com"],
            [("010-9876-5432", PiiType.PHONE), ("hong@example.com", PiiType.EMAIL)],
        ),
    ],
)
def test_overlap_keeps_one_by_priority(texts, expected):
    lines = make_lines(texts)
    assert [(texts[s.line_id][s.start : s.end], s.type) for s in detector.detect(lines)] == expected


TRICKY_LINES = [
    "계좌 010-1234-5678",
    "예금주 900101-1234567 010-1234-5678",
    "O1O-l234-5678 010-1234-5678",
    "|010-1234-5678|",
    "카드 4111-1111-1111-1111 / 1234-56**-****-7890",
    "입금 11-12-345678-90 110-123-456789",
    "01012345678@example.com",
]


@pytest.mark.parametrize("text", TRICKY_LINES)
def test_spans_never_overlap(text):
    spans = detector.detect([TextLine(id=0, text=text)])
    for a, b in itertools.pairwise(spans):
        assert a.end <= b.start


# --- 샘플 문서 --------------------------------------------------------------------

# 샘플 문서(`conftest.SAMPLE_DOCUMENTS`)에서 regex가 잡아야 하는 것 전부.
# 송장의 운송장번호, 영수증의 사업자번호·승인번호, 이름·주소는 잡지 않는다
EXPECTED_IN_SAMPLES = {
    "id_card": [(2, "900101-1234567", PiiType.RESIDENT_ID)],
    "delivery_label": [
        (3, "010-1234-5678", PiiType.PHONE),
        (5, "010-9876-5432", PiiType.PHONE),
    ],
    "messenger": [
        (2, "123456-78-901234", PiiType.ACCOUNT),
        (4, "010 2345 6789", PiiType.PHONE),
    ],
    "receipt": [(3, "1234-56**-****-7890", PiiType.CARD)],
    "bank_notice": [
        (2, "110-123-456789", PiiType.ACCOUNT),
        (4, "help@example.com", PiiType.EMAIL),
    ],
}


@pytest.mark.parametrize("document", EXPECTED_IN_SAMPLES)
def test_sample_documents(sample_lines, document):
    lines = sample_lines[document]
    texts = {line.id: line.text for line in lines}
    spans = detector.detect(lines)
    assert [
        (s.line_id, texts[s.line_id][s.start : s.end], s.type) for s in spans
    ] == EXPECTED_IN_SAMPLES[document]


def test_spans_are_sorted_by_line_and_position(sample_lines):
    for lines in sample_lines.values():
        spans = detector.detect(lines)
        assert spans == sorted(spans, key=lambda s: (s.line_id, s.start))
