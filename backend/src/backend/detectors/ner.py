"""한국어 NER 감지기. 문맥을 보고 이름(`PiiType.NAME`)과 주소(`PiiType.ADDRESS`)를 찾는다.

NER 담당자가 이 파일만 구현하면 된다. 지금은 스텁이라 아무것도 찾지 않는다.

- 입력: 위에서 아래 순서의 줄 목록. 문맥이 필요하면 앞뒤 줄을 같이 보면 된다
- 출력: `PiiSpan` 목록. `start`·`end`는 `line.text` 기준 문자 위치, `source`는 `self.name`
- 모델 로딩은 `__init__`에서 한 번만 한다 (서버 시작 시 호출). 요청마다 로딩하지 않는다
- 외부 네트워크를 호출하지 않는다. 모델 파일은 미리 받아 둔다
- 의존성(transformers 등)은 `pyproject.toml`의 optional 그룹 `ner`에 넣는다.
  이 모듈은 `GARIGAE_DETECTORS`에 `ner`가 있을 때만 import되므로, 무거운 import는 파일 맨 위에 둬도 된다
- 테스트: `tests/test_ner.py`
"""

from collections.abc import Sequence

from backend.detectors.base import PiiSpan, TextLine


class NerDetector:
    name = "ner"

    def __init__(self) -> None:
        pass

    def detect(self, lines: Sequence[TextLine]) -> list[PiiSpan]:
        return []
