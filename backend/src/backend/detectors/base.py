"""감지기 인터페이스.

감지기는 OCR이 읽은 줄 텍스트(`TextLine`)만 보고 민감정보의 문자 구간(`PiiSpan`)을 돌려준다.
이미지 좌표는 몰라도 된다. 문자 구간을 이미지 좌표로 바꾸는 일은 파이프라인이 맡는다.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, Self

from pydantic import BaseModel, Field, model_validator


class PiiType(StrEnum):
    RESIDENT_ID = "resident_id"
    FOREIGNER_ID = "foreigner_id"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    PHONE = "phone"
    EMAIL = "email"
    CARD = "card"
    ACCOUNT = "account"
    NAME = "name"
    ADDRESS = "address"

    @property
    def label(self) -> str:
        """화면에 보여줄 한국어 이름."""
        return _LABELS[self]


_LABELS = {
    PiiType.RESIDENT_ID: "주민등록번호",
    PiiType.FOREIGNER_ID: "외국인등록번호",
    PiiType.PASSPORT: "여권번호",
    PiiType.DRIVER_LICENSE: "운전면허번호",
    PiiType.PHONE: "전화번호",
    PiiType.EMAIL: "이메일",
    PiiType.CARD: "카드번호",
    PiiType.ACCOUNT: "계좌번호",
    PiiType.NAME: "이름",
    PiiType.ADDRESS: "주소",
}


class TextLine(BaseModel):
    """OCR 글자 박스를 줄 단위로 이어 붙인 텍스트.

    감지기에는 위에서 아래로 읽는 순서로 들어오고, `id`는 그 순서(0부터)다.
    그래서 `id - 1`이 바로 윗줄이다.
    """

    id: int
    text: str


class PiiSpan(BaseModel):
    """한 줄 안에서 민감정보가 있는 문자 구간."""

    line_id: int
    # `TextLine.text` 기준 문자 위치. start 포함, end 미포함 (`text[start:end]`)
    start: int = Field(ge=0)
    end: int
    type: PiiType
    # 감지기가 매긴 확신 정도
    score: float = Field(ge=0, le=1)
    # 감지기 이름 (`Detector.name`)
    source: str

    @model_validator(mode="after")
    def _check_range(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end는 start보다 커야 함")
        return self


class Detector(Protocol):
    """감지기 공통 인터페이스.

    - 모델 로딩 같은 무거운 준비는 `__init__`에서 한 번만 한다. 서버 시작 시 한 번 만들어 계속 쓴다
    - 외부 네트워크를 호출하지 않는다
    - 찾은 게 없으면 빈 목록을 돌려준다
    """

    name: str

    def detect(self, lines: Sequence[TextLine]) -> list[PiiSpan]: ...
