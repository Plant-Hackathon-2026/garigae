"""이름·주소 규칙 감지기. 라벨(`성명`, `주소` 등)과 주소 패턴으로 찾는다.

지금은 스텁이다. #10에서 구현한다.
"""

from collections.abc import Sequence

from backend.detectors.base import PiiSpan, TextLine


class RulesDetector:
    name = "rules"

    def detect(self, lines: Sequence[TextLine]) -> list[PiiSpan]:
        return []
