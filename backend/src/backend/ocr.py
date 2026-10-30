"""OCR 엔진. 이미지에서 글자 박스(꼭짓점 4개, 텍스트, 신뢰도)를 읽는다.

파이프라인은 `OcrEngine` 프로토콜만 보므로, 테스트에서는 가짜 엔진으로 바꿔 끼워 torch 없이 돌린다.
"""

import threading
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from backend.config import Settings

Point = tuple[float, float]


@dataclass(frozen=True)
class OcrBox:
    """OCR이 읽은 글자 박스 하나."""

    # 꼭짓점 4개, 시계 방향. EasyOCR는 x+y가 가장 작은 점부터 준다.
    # 45° 넘게 기울면 첫 점이 글자 기준 왼쪽 위가 아니므로 글자 방향은 lines에서 다시 정한다
    points: tuple[Point, Point, Point, Point]
    text: str
    confidence: float


class OcrEngine(Protocol):
    @property
    def device(self) -> str:
        """추론 장치: cuda, mps, cpu."""
        ...

    def read(self, image: np.ndarray) -> list[OcrBox]:
        """RGB uint8 `(높이, 너비, 3)` 이미지에서 글자 박스를 읽는다."""
        ...


class EasyOcrEngine:
    """EasyOCR `Reader` 하나를 락으로 직렬화해서 쓴다.

    모델 로딩이 몇 초 걸리므로 서버 시작 시 한 번만 만든다. 요청은 스레드풀에서 동시에 들어오지만
    Reader 하나를 여러 스레드가 같이 부르지 않고, GPU 메모리도 한 장 분량만 쓰도록 추론은 한 번에 하나씩 한다.
    """

    def __init__(self, settings: Settings) -> None:
        # torch를 끌고 와서 무겁다. 가짜 엔진을 쓰는 테스트에서는 import하지 않는다
        import easyocr

        self._reader = easyocr.Reader(
            settings.ocr_languages,
            gpu=settings.ocr_gpu,
            model_storage_directory=str(settings.ocr_model_dir) if settings.ocr_model_dir else None,
            download_enabled=False,
            verbose=False,
        )
        self._rotation_info = list(settings.ocr_rotations) or None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return str(self._reader.device)

    def warmup(self) -> None:
        """작은 이미지로 한 번 돌려 둔다. 첫 추론의 장치 초기화 지연을 첫 요청이 떠안지 않게 한다."""
        image = np.full((64, 320, 3), 255, np.uint8)
        # 글자 인식까지 돌도록 글자를 그려 둔다 (빈 이미지는 글자 영역 검출에서 끝난다)
        cv2.putText(image, "0123 4567", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        self.read(image)

    def read(self, image: np.ndarray) -> list[OcrBox]:
        # `readtext`에 배열을 넘기면 3채널을 BGR로 보고 흑백으로 바꾼다. 글자 영역 검출(CRAFT)은 RGB를,
        # 글자 인식은 흑백을 받으므로 `readtext`가 하는 두 단계를 직접 불러 각각 맞는 입력을 준다
        grey = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        with self._lock:
            horizontal, free = self._reader.detect(image)
            horizontal, free = horizontal[0], free[0]
            if not horizontal and not free:
                return []
            results = self._reader.recognize(
                grey, horizontal, free, rotation_info=self._rotation_info, reformat=False
            )
        return [
            OcrBox(
                points=tuple((float(x), float(y)) for x, y in box),
                text=str(text),
                confidence=float(confidence),
            )
            for box, text, confidence in results
        ]
