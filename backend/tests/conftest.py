import pytest

from backend.detectors.base import TextLine

# 가짜 데이터로 만든 OCR 줄 텍스트 샘플. 이미지 종류별로 OCR이 줄 단위로 읽어 낸 모양을 흉내 냈다.
# 감지기 테스트와 NER 개발에 쓴다. 실제 개인정보는 넣지 않는다.
SAMPLE_DOCUMENTS: dict[str, list[str]] = {
    "id_card": [
        "주민등록증",
        "홍길동 (洪吉童)",
        "900101-1234567",
        "서울특별시 가상구 테스트로 12",
        "2020.1.1.",
        "서울특별시 가상구청장",
    ],
    "delivery_label": [
        "택배 운송장",
        "운송장번호 1234-5678-9012",
        "받는 분: 김철수",
        "연락처: 010-1234-5678",
        "주소: 부산광역시 해운대구 가상로 34, 101동 202호",
        "보내는 분: 이영희 010-9876-5432",
    ],
    "messenger": [
        "엄마",
        "계좌번호 알려줄게",
        "국민은행 123456-78-901234 홍길동",
        "오늘 7시에 강남역에서 봐",
        "내 번호 바뀌었어 010 2345 6789",
    ],
    "receipt": [
        "가상마트 강남점",
        "사업자번호 123-45-67890",
        "대표 박민수",
        "카드번호 1234-56**-****-7890",
        "승인번호 12345678",
        "합계 32,000원",
    ],
    "bank_notice": [
        "입금 계좌 안내",
        "은행: 신한은행",
        "계좌: 110-123-456789",
        "예금주: 최지우",
        "문의: help@example.com",
    ],
}


def make_lines(texts: list[str]) -> list[TextLine]:
    return [TextLine(id=i, text=text) for i, text in enumerate(texts)]


@pytest.fixture
def sample_lines() -> dict[str, list[TextLine]]:
    """`SAMPLE_DOCUMENTS`를 `TextLine` 목록으로 바꾼 것."""
    return {name: make_lines(texts) for name, texts in SAMPLE_DOCUMENTS.items()}
