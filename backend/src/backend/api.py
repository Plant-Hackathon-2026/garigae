"""API 라우터.

`/api/detect`, `/api/redact`, `/api/redact/auto`는 아직 스텁이다. 요청 형식은 실제와 같고,
감지는 고정 예시를, 가리기는 원본 이미지를 그대로 돌려준다. 프론트 연동을 먼저 시작하기 위한 것이다.
실제 처리는 #7(`/api/redact/auto`), #9(`/api/detect`, `/api/redact`)에서 붙인다.
"""

import io
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import ValidationError

from backend.detectors.base import PiiType
from backend.schemas import (
    DETECT_EXAMPLE,
    DetectResponse,
    HealthResponse,
    PiiTypeInfo,
    RedactMode,
    RegionList,
)

router = APIRouter(prefix="/api")

_IMAGE_RESPONSE = {
    200: {
        "content": {"image/jpeg": {}, "image/png": {}, "image/webp": {}},
        "description": "가린 이미지. 입력과 같은 형식",
    }
}
_MEDIA_TYPES = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="healthy", ocr_ready=False, device=None)


@router.get("/pii-types", response_model=list[PiiTypeInfo])
def pii_types() -> list[PiiTypeInfo]:
    """감지 항목 종류와 한국어 이름."""
    return [PiiTypeInfo(type=pii_type, label=pii_type.label) for pii_type in PiiType]


@router.post("/detect", response_model=DetectResponse)
def detect(file: Annotated[UploadFile, File(description="JPEG/PNG/WEBP, 최대 10MB")]) -> DetectResponse:
    """민감정보 영역과 OCR 글자 박스를 찾는다. (스텁: 예시 좌표를 업로드한 이미지 크기에 맞춰 돌려줌)"""
    _, image_format, (width, height) = _read_image(file)
    sx = width / DETECT_EXAMPLE["image"]["width"]
    sy = height / DETECT_EXAMPLE["image"]["height"]

    def scale(item: dict) -> dict:
        return {**item, "polygon": [[x * sx, y * sy] for x, y in item["polygon"]]}

    return DetectResponse.model_validate(
        {
            **DETECT_EXAMPLE,
            "image": {"width": width, "height": height, "format": image_format.lower()},
            "detections": [scale(d) for d in DETECT_EXAMPLE["detections"]],
            "text_regions": [scale(t) for t in DETECT_EXAMPLE["text_regions"]],
        }
    )


@router.post("/redact", response_class=Response, responses=_IMAGE_RESPONSE)
def redact(
    file: Annotated[UploadFile, File(description="원본 이미지 (서버가 저장하지 않으므로 다시 보냄)")],
    regions: Annotated[str, Form(description='가릴 영역 JSON. 예: [{"polygon": [[x, y], ...]}]')],
    mode: Annotated[RedactMode, Form()] = "blur",
) -> Response:
    """보낸 영역을 가린 이미지를 돌려준다. (스텁: 원본을 그대로 돌려줌)"""
    data, image_format, _ = _read_image(file)
    try:
        RegionList.validate_json(regions)
    except ValidationError as error:
        detail = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or 'regions'}: {e['msg']}" for e in error.errors()
        )
        raise HTTPException(400, f"regions 형식 오류: {detail}") from None
    return Response(data, media_type=_MEDIA_TYPES[image_format])


@router.post("/redact/auto", response_class=Response, responses=_IMAGE_RESPONSE)
def redact_auto(
    file: Annotated[UploadFile, File(description="JPEG/PNG/WEBP, 최대 10MB")],
    mode: Annotated[RedactMode, Form()] = "blur",
    types: Annotated[str | None, Form(description="가릴 항목, 쉼표 구분. 생략하면 전체")] = None,
) -> Response:
    """감지된 영역을 전부 가린 이미지를 돌려준다. (스텁: 원본을 그대로 돌려줌)"""
    data, image_format, _ = _read_image(file)
    for name in filter(None, (name.strip() for name in (types or "").split(","))):
        try:
            PiiType(name)
        except ValueError:
            raise HTTPException(400, f"알 수 없는 항목: {name}") from None
    return Response(data, media_type=_MEDIA_TYPES[image_format])


def _read_image(file: UploadFile) -> tuple[bytes, str, tuple[int, int]]:
    """원본 바이트, 형식, EXIF 회전 정보를 적용한 (가로, 세로)."""
    # 스텁용 최소 검사. 크기 제한과 메타데이터 처리는 #9의 image_io로 옮긴다
    data = file.file.read()
    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise HTTPException(400, "이미지를 열 수 없음") from None
    if image.format not in _MEDIA_TYPES:
        raise HTTPException(415, "JPEG/PNG/WEBP만 지원")
    width, height = image.size
    # 5~8은 90°·270° 회전이라 가로·세로가 바뀐다
    if image.getexif().get(ExifTags.Base.Orientation, 1) in (5, 6, 7, 8):
        width, height = height, width
    return data, image.format, (width, height)
