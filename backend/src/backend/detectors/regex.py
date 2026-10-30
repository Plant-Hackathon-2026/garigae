"""정규식 감지기. 형식이 정해진 번호(식별번호, 연락처, 금융정보)를 찾는다.

줄마다 이렇게 처리한다.

1. 항목별 정규식을 돌리고 검증(날짜, Luhn)을 통과한 것만 후보로 남긴다
2. 같은 구간을 여러 항목이 잡으면 우선순위(`_RULES` 순서)가 앞선 하나만 남긴다

감지 기준은 `plans/backend-plan.md`의 "감지 규칙 > 정규식 감지기" 표를 따른다.
"""

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

from backend.detectors.base import PiiSpan, PiiType, TextLine

# score 기준. 형식 검증까지 통과하면 높게, 형식만 맞거나 키워드 조건으로만 잡으면 낮게 준다.
# 파이프라인이 여기에 OCR 신뢰도를 곱해 응답의 confidence로 쓴다.
# 검증 통과: 주민·외국인등록번호(날짜), 카드번호(Luhn), 전화번호(국번·자릿수)
SCORE_VALIDATED = 0.9
# 이메일은 별도 검증이 없지만 `@`와 도메인 형식 자체가 분명하다
SCORE_EMAIL = 0.9
# 일부가 가려져 Luhn을 볼 수 없음
SCORE_MASKED_CARD = 0.75

_DIGITS = frozenset("0123456789")

# --- 패턴 ---------------------------------------------------------------------
# 모든 패턴은 re.ASCII로 컴파일한다. `\d`는 0-9, `\s`는 ASCII 공백만 뜻한다.

# 숫자 경계. 앞뒤가 숫자이거나 `-숫자`로 이어지면 더 긴 번호(송장·주문번호 등)의 일부로 본다
_START = r"(?<!\d)(?<!\d-)"
_END = r"(?!-?\d)"
# 번호 사이 구분자: 하이픈, 공백 하나, 없음.
# 줄 재구성이 OCR 박스를 공백으로 이어 붙이므로 하이픈 앞뒤 공백 하나씩도 허용한다 (`010 -1234`)
_SEP = r"(?:\s?-\s?|\s)?"


def _compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, re.ASCII | flags)


# 주민·외국인등록번호: YYMMDD + 구분자 + 뒷자리 첫 숫자(주민 1-4, 외국인 5-8) + 6자리
_RESIDENT = _compile(
    rf"{_START}(?P<yy>\d{{2}})(?P<mm>\d{{2}})(?P<dd>\d{{2}}){_SEP}(?P<gender>[1-8])\d{{6}}{_END}"
)

# 카드번호: 4-4-4-4(16자리), 아멕스 4-6-5(15자리, 34·37로 시작)
_CARD_16 = rf"\d{{4}}(?:{_SEP}\d{{4}}){{3}}"
_CARD_AMEX = rf"3[47]\d{{2}}{_SEP}\d{{6}}{_SEP}\d{{5}}"
_CARD = _compile(rf"{_START}(?:{_CARD_16}|{_CARD_AMEX}){_END}")

# 일부가 가려진 카드번호 (`1234-56**-****-7890`). 영수증에서 흔한 가림 문자를 받는다
_MASK_CHARS = frozenset("*＊xX#●•")
_MASKABLE = r"[\d*＊xX#●•]"
_MASKED_16 = rf"{_MASKABLE}{{4}}(?:{_SEP}{_MASKABLE}{{4}}){{3}}"
_MASKED_AMEX = rf"{_MASKABLE}{{4}}{_SEP}{_MASKABLE}{{6}}{_SEP}{_MASKABLE}{{5}}"
_MASKED_CARD = _compile(
    rf"(?<![0-9A-Za-z*＊#●•])(?<!\d-)(?:{_MASKED_16}|{_MASKED_AMEX})"
    rf"(?![0-9A-Za-z*＊#●•])(?!-{_MASKABLE})"
)

# 전화번호: 국번 + 3~4자리 + 4자리. `(02) 1234-5678`, `02)1234-5678`처럼 괄호 쓴 국번도 받는다
# - 휴대폰 010·011·016·017·018·019
# - 유선 02(서울), 031-033, 041-044, 051-055, 061-064 (계획의 `0[3-6][1-5]` 중 실제 지역번호만)
# - 인터넷 전화 070, 안심번호 050X
# 대표번호(15XX·16XX·18XX)는 개인 연락처가 아니라 받지 않는다. 0으로 시작하지 않아서 위 국번에
# 걸리지 않는다. 080(수신자 부담)도 업체 번호라 넣지 않았다
_PHONE_PREFIX = r"(?:01[016789]|02|03[1-3]|04[1-4]|05[1-5]|06[1-4]|070|050\d)"
_PHONE_SEP = r"(?:\s?[-.]\s?|\s)?"
_PHONE_AREA = rf"(?:\(\s?{_PHONE_PREFIX}\s?\)\s?|{_PHONE_PREFIX}(?:\)\s?|{_PHONE_SEP}))"
_PHONE = _compile(rf"{_START}{_PHONE_AREA}\d{{3,4}}{_PHONE_SEP}\d{{4}}{_END}")

# 이메일. 줄 재구성이 박스 사이에 넣은 공백을 생각해 `@` 앞뒤 공백 하나는 허용한다
_EMAIL = _compile(
    r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9][A-Za-z0-9._%+-]*\s?@\s?"
    r"[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![A-Za-z0-9-])"
)


# --- 검증 ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    line: TextLine


_Verdict = tuple[PiiType, float] | None


def _judge_resident(match: re.Match[str], ctx: _Context) -> _Verdict:
    # 뒷자리 첫 숫자로 출생 세기를 안다: 1·2·5·6 → 1900년대, 3·4·7·8 → 2000년대
    gender = match["gender"]
    year = (1900 if gender in "1256" else 2000) + int(match["yy"])
    if not _is_valid_date(year, int(match["mm"]), int(match["dd"])):
        return None
    # 체크섬은 보지 않는다. 2020년 10월 이후 발급 번호는 마지막 자리 규칙이 없어졌다
    pii_type = PiiType.RESIDENT_ID if gender in "1234" else PiiType.FOREIGNER_ID
    return pii_type, SCORE_VALIDATED


def _is_valid_date(year: int, month: int, day: int) -> bool:
    try:
        date(year, month, day)  # 윤년도 여기서 걸러진다
    except ValueError:
        return False
    return True


def _judge_card(match: re.Match[str], ctx: _Context) -> _Verdict:
    if not _luhn_ok(_digits_of(match.group())):
        return None
    return PiiType.CARD, SCORE_VALIDATED


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _judge_masked_card(match: re.Match[str], ctx: _Context) -> _Verdict:
    text = match.group()
    # 가림 문자가 없으면 `_CARD`가 맡을 몫이고, 숫자가 하나도 없으면 가릴 정보가 없다
    if not _MASK_CHARS.intersection(text) or not _DIGITS.intersection(text):
        return None
    return PiiType.CARD, SCORE_MASKED_CARD


def _judge_phone(match: re.Match[str], ctx: _Context) -> _Verdict:
    # 국번과 자릿수는 패턴이 이미 확인했다
    return PiiType.PHONE, SCORE_VALIDATED


def _judge_email(match: re.Match[str], ctx: _Context) -> _Verdict:
    return PiiType.EMAIL, SCORE_EMAIL


def _digits_of(text: str) -> str:
    return "".join(ch for ch in text if ch in _DIGITS)


# --- 규칙과 우선순위 ------------------------------------------------------------


@dataclass(frozen=True)
class _Rule:
    pattern: re.Pattern[str]
    judge: Callable[[re.Match[str], _Context], _Verdict]


# 같은 구간을 여러 항목이 잡으면 이 목록에서 앞선 규칙 하나만 남긴다.
# 형식이 독특하고 검증이 강한 순서로 둬서, 틀릴 가능성이 낮은 쪽의 종류를 보여 준다.
_RULES: tuple[_Rule, ...] = (
    # 이메일: `@`와 도메인이 분명하다. 아이디 자리에 전화번호가 있어도(`01012345678@...`) 전체가
    # 이메일이고, 전화번호를 먼저 두면 도메인 부분이 안 가려진다
    _Rule(_EMAIL, _judge_email),
    # 주민·외국인등록번호: 13자리 + 날짜 검사
    _Rule(_RESIDENT, _judge_resident),
    # 카드번호: Luhn 검사
    _Rule(_CARD, _judge_card),
    # 전화번호: 국번·자릿수 검사
    _Rule(_PHONE, _judge_phone),
    # 가려진 카드번호: Luhn을 볼 수 없다
    _Rule(_MASKED_CARD, _judge_masked_card),
)


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    type: PiiType
    score: float
    rank: int  # `_RULES` 안의 순서. 작을수록 앞선다


def _candidates(ctx: _Context) -> list[_Candidate]:
    found: list[_Candidate] = []
    for rank, rule in enumerate(_RULES):
        for match in rule.pattern.finditer(ctx.line.text):
            verdict = rule.judge(match, ctx)
            if verdict is None:
                continue
            pii_type, score = verdict
            start, end = match.span()
            found.append(_Candidate(start, end, pii_type, score, rank))
    return found


def _resolve(candidates: list[_Candidate]) -> list[_Candidate]:
    """겹치는 후보 중 우선순위가 앞선 것만 남긴다."""
    chosen: list[_Candidate] = []
    for cand in sorted(candidates, key=lambda c: (c.rank, c.start)):
        if all(cand.end <= other.start or other.end <= cand.start for other in chosen):
            chosen.append(cand)
    return sorted(chosen, key=lambda c: c.start)


class RegexDetector:
    name = "regex"

    def detect(self, lines: Sequence[TextLine]) -> list[PiiSpan]:
        return [
            PiiSpan(
                line_id=line.id,
                start=cand.start,
                end=cand.end,
                type=cand.type,
                score=cand.score,
                source=self.name,
            )
            for line in lines
            for cand in _resolve(_candidates(_Context(line)))
        ]
