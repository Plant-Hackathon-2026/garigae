"""정규식 감지기. 형식이 정해진 번호(식별번호, 연락처, 금융정보)를 찾는다.

지금은 휴대폰 번호만 있다. 나머지 항목과 검증 로직은 #8에서 채운다.
"""

import re
from collections.abc import Sequence

from backend.detectors.base import PiiSpan, PiiType, TextLine

_MOBILE = re.compile(r"(?<!\d)01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d)")


class RegexDetector:
    name = "regex"

    def detect(self, lines: Sequence[TextLine]) -> list[PiiSpan]:
        return [
            PiiSpan(
                line_id=line.id,
                start=match.start(),
                end=match.end(),
                type=PiiType.PHONE,
                score=0.9,
                source=self.name,
            )
            for line in lines
            for match in _MOBILE.finditer(line.text)
        ]
