"""설정(`GARIGAE_DETECTORS`)에 따라 감지기를 만든다."""

from collections.abc import Sequence

from backend.config import DetectorName
from backend.detectors.base import Detector


def build_detectors(names: Sequence[DetectorName]) -> list[Detector]:
    return [_create(name) for name in dict.fromkeys(names)]


def _create(name: DetectorName) -> Detector:
    # 켠 감지기만 import한다. NER 의존성이 없어도 ner를 끄면 서버가 뜬다
    if name == "regex":
        from backend.detectors.regex import RegexDetector

        return RegexDetector()
    if name == "rules":
        from backend.detectors.rules import RulesDetector

        return RulesDetector()
    if name == "ner":
        from backend.detectors.ner import NerDetector

        return NerDetector()
    raise ValueError(f"알 수 없는 감지기: {name}")
