"""서버 설정. 환경변수 `GARIGAE_*`로 바꾼다.

목록 값은 쉼표로 구분한다. 예: `GARIGAE_DETECTORS=regex,rules,ner`
"""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

OcrDevice = Literal["auto", "cuda", "mps", "cpu"]
OcrRotation = Literal[90, 180, 270]
DetectorName = Literal["regex", "rules", "ner"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GARIGAE_")

    # OCR
    ocr_languages: Annotated[list[str], NoDecode] = ["ko", "en"]
    # auto: CUDA → MPS → CPU 순으로 고른다
    ocr_device: OcrDevice = "auto"
    # 비워 두면 EasyOCR 기본 위치(~/.EasyOCR/model)
    ocr_model_dir: Path | None = None
    # 90°·180° 돌아간 이미지 인식 (예: 90,180,270). 회전 수만큼 느려지므로 기본은 끔
    ocr_rotations: Annotated[list[OcrRotation], NoDecode] = []

    # 업로드 제한
    max_upload_bytes: int = 10 * 1024 * 1024
    # 압축 폭탄 방지
    max_image_pixels: int = 40_000_000

    cors_origins: Annotated[list[str], NoDecode] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # 켤 감지기. ner는 NER 구현이 들어온 뒤 켠다
    detectors: Annotated[list[DetectorName], NoDecode] = ["regex", "rules"]

    @field_validator("ocr_languages", "cors_origins", "detectors", mode="before")
    @classmethod
    def _split_comma(cls, value: object) -> object:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("ocr_rotations", mode="before")
    @classmethod
    def _split_comma_int(cls, value: object) -> object:
        if isinstance(value, str):
            return [int(item) for item in value.split(",") if item.strip()]
        return value

    @property
    def ocr_gpu(self) -> bool | str:
        """EasyOCR `Reader`의 `gpu` 인자. True면 EasyOCR가 CUDA → MPS → CPU 순으로 고른다."""
        if self.ocr_device == "auto":
            return True
        if self.ocr_device == "cpu":
            return False
        return self.ocr_device


@lru_cache
def get_settings() -> Settings:
    return Settings()
