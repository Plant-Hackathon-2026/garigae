"""OCR → 줄 재구성 → 감지기 → 구간 합치기 → 좌표.

로그에는 처리 시간과 개수만 남긴다. OCR 텍스트, 감지된 문자열, 이미지 내용은 남기지 않는다.
"""

import logging
import time
from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from backend import geometry
from backend.detectors.base import Detector, PiiSpan, PiiType
from backend.lines import LineLayout, build_lines
from backend.ocr import OcrEngine
from backend.schemas import Detection, TextRegion

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    detections: list[Detection]
    # OCR이 읽은 모든 글자 박스 (줄 순서)
    text_regions: list[TextRegion]


class Pipeline:
    """서버 시작 시 한 번 만들어 계속 쓴다. 요청마다 `run`을 부른다."""

    def __init__(self, engine: OcrEngine, detectors: Sequence[Detector]) -> None:
        self.engine = engine
        self.detectors = list(detectors)

    @property
    def device(self) -> str:
        return self.engine.device

    def run(self, image: np.ndarray, types: Collection[PiiType] | None = None) -> PipelineResult:
        """RGB uint8 `(높이, 너비, 3)` 이미지에서 민감정보 영역을 찾는다. `types`를 주면 그 항목만."""
        height, width = image.shape[:2]
        size = (width, height)

        started = time.perf_counter()
        boxes = self.engine.read(image)
        ocr_ms = (time.perf_counter() - started) * 1000

        layouts = build_lines(boxes)
        lines = [layout.line for layout in layouts]
        spans = [span for detector in self.detectors for span in detector.detect(lines)]
        valid = [span for span in spans if _inside(span, layouts)]
        if len(valid) != len(spans):
            logger.warning("줄 범위를 벗어난 감지 구간 %d개를 버림", len(spans) - len(valid))
        if types is not None:
            wanted = set(types)
            valid = [span for span in valid if span.type in wanted]

        detections = []
        for span in merge_spans(valid):
            layout = layouts[span.line_id]
            pieces = layout.pieces(span.start, span.end)
            if not pieces:  # 박스 사이 공백만 가리킴
                continue
            ocr_confidence = min(box.confidence for box in layout.boxes_in(span.start, span.end))
            detections.append(
                Detection(
                    id=f"d{len(detections) + 1}",
                    type=span.type,
                    label=span.type.label,
                    text=layout.line.text[span.start : span.end],
                    polygon=geometry.to_points(geometry.span_polygon(pieces, size)),
                    confidence=_clamp(ocr_confidence * span.score),
                    source=span.source,
                )
            )

        text_regions = [
            TextRegion(
                id=f"t{index + 1}",
                text=box.text,
                polygon=geometry.to_points(geometry.clip(box.quad, size)),
                confidence=_clamp(box.confidence),
            )
            for index, box in enumerate(box for layout in layouts for box in layout.boxes)
        ]

        logger.info(
            "pipeline: %dx%d, ocr %.0fms, total %.0fms, boxes=%d, lines=%d, spans=%d, detections=%d",
            width,
            height,
            ocr_ms,
            (time.perf_counter() - started) * 1000,
            len(boxes),
            len(layouts),
            len(valid),
            len(detections),
        )
        return PipelineResult(detections=detections, text_regions=text_regions)


def merge_spans(spans: Sequence[PiiSpan]) -> list[PiiSpan]:
    """같은 줄에서 겹치는 구간을 하나로 합친다 (겹치면 넓게).

    종류가 달라도 겹치면 합친다. 같은 글자를 두 영역으로 나눠 보여주지 않기 위해서다.
    합친 구간의 종류·감지기·점수는 점수가 가장 높은 구간을 따른다 (같으면 먼저 나온 감지기).
    맞닿기만 한 구간(`end == start`)은 합치지 않는다. 결과는 (줄, 시작 위치) 순서다.
    """
    ordered = sorted(enumerate(spans), key=lambda pair: (pair[1].line_id, pair[1].start, -pair[1].end))
    merged: list[tuple[PiiSpan, int]] = []  # (구간, 대표 구간의 원래 순서)
    for index, span in ordered:
        if merged and merged[-1][0].line_id == span.line_id and span.start < merged[-1][0].end:
            current, current_index = merged[-1]
            best, best_index = (
                (span, index)
                if (span.score, -index) > (current.score, -current_index)
                else (current, current_index)
            )
            merged[-1] = (
                best.model_copy(update={"start": current.start, "end": max(current.end, span.end)}),
                best_index,
            )
        else:
            merged.append((span, index))
    return [span for span, _ in merged]


def _inside(span: PiiSpan, layouts: Sequence[LineLayout]) -> bool:
    return 0 <= span.line_id < len(layouts) and span.end <= len(layouts[span.line_id].line.text)


def _clamp(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)
