"""API 요청·응답 모델.

좌표는 EXIF 회전 정보를 적용한 원본 이미지의 픽셀 기준이다. 브라우저가 보여주는 방향과 같다.
영역은 꼭짓점 4개짜리 `polygon` 하나로만 주고받는다.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, TypeAdapter

from backend.detectors.base import PiiType

Point = tuple[FiniteFloat, FiniteFloat]
# 꼭짓점 4개, 시계 방향. 대부분 회전된 직사각형이고, 기울기가 약 5.7° 미만이면 수평 직사각형이다.
# 직사각형이라고 가정하지 말고 꼭짓점을 그대로 이어 그린다
Polygon = Annotated[list[Point], Field(min_length=4, max_length=4)]

ImageFormat = Literal["jpeg", "png", "webp"]
RedactMode = Literal["blur", "pixelate", "box"]


class ImageInfo(BaseModel):
    width: int
    height: int
    format: ImageFormat


class Detection(BaseModel):
    id: str
    type: PiiType
    label: str = Field(description="항목의 한국어 이름")
    text: str = Field(description="감지된 문자열")
    polygon: Polygon
    confidence: float = Field(ge=0, le=1, description="OCR 신뢰도 × 감지기 점수")
    source: str = Field(description="찾은 감지기: regex, rules, ner")


class TextRegion(BaseModel):
    id: str
    text: str
    polygon: Polygon
    confidence: float = Field(ge=0, le=1, description="OCR 신뢰도")


DETECT_EXAMPLE = {
    "image": {"width": 1280, "height": 960, "format": "jpeg"},
    "detections": [
        {
            "id": "d1",
            "type": "phone",
            "label": "전화번호",
            "text": "010-0000-0000",
            "polygon": [[412, 233], [580, 233], [580, 261], [412, 261]],
            "confidence": 0.93,
            "source": "regex",
        },
        {
            "id": "d2",
            "type": "name",
            "label": "이름",
            "text": "홍길동",
            "polygon": [[452, 146], [548, 160], [545, 195], [449, 181]],
            "confidence": 0.71,
            "source": "rules",
        },
    ],
    "text_regions": [
        {
            "id": "t1",
            "text": "받는 분: 홍길동",
            "polygon": [[340, 130], [544, 160], [541, 194], [337, 164]],
            "confidence": 0.88,
        },
        {
            "id": "t2",
            "text": "연락처: 010-0000-0000",
            "polygon": [[340, 231], [584, 231], [584, 263], [340, 263]],
            "confidence": 0.95,
        },
    ],
    "elapsed_ms": 820,
}


class DetectResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [DETECT_EXAMPLE]})

    image: ImageInfo
    detections: list[Detection] = Field(description="민감정보로 감지된 영역. 기본으로 켜 둔다")
    text_regions: list[TextRegion] = Field(
        description="OCR이 읽은 모든 글자 박스. 감지가 놓친 부분을 사용자가 눌러서 추가로 가린다"
    )
    elapsed_ms: int


class Region(BaseModel):
    """가릴 영역. 사용자가 그린 사각형은 `[[x, y], [x+w, y], [x+w, y+h], [x, y+h]]`로 바꿔 보낸다."""

    polygon: Polygon


# `/api/redact`의 `regions` 폼 필드(JSON 문자열) 파서
RegionList = TypeAdapter(list[Region])


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    ocr_ready: bool
    device: str | None = Field(description="OCR 장치: cuda, mps, cpu. 모델 로딩 전이면 null")


class PiiTypeInfo(BaseModel):
    type: PiiType
    label: str
