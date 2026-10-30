"""정규식 감지기. 형식이 정해진 번호(식별번호, 연락처, 금융정보)를 찾는다.

줄마다 이렇게 처리한다.

1. 항목별 정규식을 돌리고 검증(날짜, Luhn, 지역 코드, 키워드)을 통과한 것만 후보로 남긴다
2. 같은 구간을 여러 항목이 잡으면 우선순위(`_RULES` 순서)가 앞선 하나만 남긴다

감지 기준은 `plans/backend-plan.md`의 "감지 규칙 > 정규식 감지기" 표를 따른다.
"""

import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from backend.detectors.base import PiiSpan, PiiType, TextLine

# score 기준. 형식 검증까지 통과하면 높게, 형식만 맞거나 키워드 조건으로만 잡으면 낮게 준다.
# 파이프라인이 여기에 OCR 신뢰도를 곱해 응답의 confidence로 쓴다.
# 검증 통과: 주민·외국인등록번호(날짜), 카드번호(Luhn), 운전면허번호(지역 코드),
# 전화번호(국번·자릿수)
SCORE_VALIDATED = 0.9
# 이메일은 별도 검증이 없지만 `@`와 도메인 형식 자체가 분명하다
SCORE_EMAIL = 0.9
SCORE_PASSPORT_WITH_KEYWORD = 0.9
# 여권번호 형식만 맞음. 영문 1자 + 숫자 8자리라 제품·주문 코드와 헷갈릴 수 있다
SCORE_PASSPORT = 0.6
# 일부가 가려져 Luhn을 볼 수 없음
SCORE_MASKED_CARD = 0.75
# 검증 수단이 없고 키워드 조건으로만 잡음
SCORE_ACCOUNT = 0.7

_DIGITS = frozenset("0123456789")

# --- 패턴 ---------------------------------------------------------------------
# 모든 패턴은 re.ASCII로 컴파일한다. `\d`는 0-9, `\s`는 ASCII 공백만 뜻한다.

# 숫자 경계. 앞뒤가 숫자이거나 `-숫자`로 이어지면 더 긴 번호(송장·주문번호 등)의 일부로 본다
_START = r"(?<!\d)(?<!\d-)"
_END = r"(?!-?\d)"
# 번호 사이 구분자: 하이픈, 공백 하나, 없음.
# 줄 재구성이 OCR 박스를 공백으로 이어 붙이므로 하이픈 앞뒤 공백 하나씩도 허용한다 (`010 -1234`)
_SEP = r"(?:\s?-\s?|\s)?"
_SEP_REQUIRED = r"(?:\s?-\s?|\s)"


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

# 운전면허번호.
# - 현행 `NN-NN-NNNNNN-NN`: 지역 코드 2 + 발급 연도 2 + 일련번호 6 + 검증·재발급 2
# - 구형 `서울 NN-NNNNNN-NN`: 지역 코드 자리에 지역명. 2014년 3분기부터 숫자 코드로 바뀌었다
# 지역 코드: 11 서울, 12 부산, 13 경기(남부), 14 강원, 15 충북, 16 충남, 17 전북, 18 전남,
# 19 경북, 20 경남, 21 제주, 22 대구, 23 인천, 24 광주, 25 대전, 26 울산, 28 경기북부.
# 27은 결번이다(세종은 면허를 따로 관할하지 않음). 경기북부(28)는 구형 지역명이 없다.
# 근거: Microsoft Purview "South Korea driver's license number" 정의(허용 코드 11-26, 28),
# 지역별 코드 표를 정리한 공개 게시글.
# 구분자가 없는 12자리는 택배 송장번호(12자리)와 겹쳐서 받지 않는다
_LICENSE_REGION_CODES = frozenset(f"{code:02d}" for code in [*range(11, 27), 28])
_LICENSE_REGION_NAMES = (
    "서울|부산|경기|강원|충북|충남|전북|전남|경북|경남|제주|대구|인천|광주|대전|울산"
)
_LICENSE_HEAD = rf"(?:(?P<code>\d{{2}}){_SEP_REQUIRED}|(?:{_LICENSE_REGION_NAMES})\s?)"
_DRIVER_LICENSE = _compile(
    rf"{_START}{_LICENSE_HEAD}\d{{2}}{_SEP_REQUIRED}\d{{6}}{_SEP_REQUIRED}\d{{2}}{_END}"
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

# 여권번호: 구형 `[MSRODG]` + 8자리, 차세대 `[MSRODG]` + 3자리 + 영문 + 4자리
_PASSPORT = _compile(r"(?<![A-Za-z0-9])[MSRODG](?:\d{8}|\d{3}[A-Z]\d{4})(?![A-Za-z0-9])")
_PASSPORT_KEYWORD = _compile(r"여권|passport", re.IGNORECASE)
# 여권 정보면은 "여권번호/Passport No." 라벨 아래 한두 줄에 번호가 오므로 위아래 2줄까지 본다
_PASSPORT_KEYWORD_LINES = 2

# 계좌번호: 하이픈으로 나뉜 그룹이 2개 이상인 숫자열 전체. 숫자 개수(10~14)는 검증에서 본다
_ACCOUNT = _compile(rf"{_START}\d+(?:\s?-\s?\d+)+{_END}")
# 은행명은 "OO은행"·"OO뱅크"·"OO금고"로 대부분 잡힌다. 줄여 쓰는 이름 중
# 우리·하나·기업·부산·대구·광주·경남·전북·제주처럼 일반 단어나 지역명과 겹치는 것은 넣지 않았다
_ACCOUNT_KEYWORD = _compile(
    r"계좌|입금|예금주"
    r"|은행|뱅크|금고|신협|수협|농협|축협|우체국|국민|신한|카뱅|케뱅|씨티"
    r"|(?<![A-Za-z])(?:KB|NH|IBK|KDB|SC)(?![A-Za-z])"
)


# --- 검증 ---------------------------------------------------------------------


@dataclass(frozen=True)
class _Context:
    line: TextLine
    # 줄 id → 원본 텍스트. 줄 id는 위→아래 순서라 `id - 1`이 바로 윗줄이다
    texts: Mapping[int, str]

    def text_at(self, offset: int) -> str:
        return self.texts.get(self.line.id + offset, "")


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


def _judge_driver_license(match: re.Match[str], ctx: _Context) -> _Verdict:
    code = match["code"]
    if code is not None and code not in _LICENSE_REGION_CODES:
        return None
    return PiiType.DRIVER_LICENSE, SCORE_VALIDATED


def _judge_phone(match: re.Match[str], ctx: _Context) -> _Verdict:
    # 국번과 자릿수는 패턴이 이미 확인했다
    return PiiType.PHONE, SCORE_VALIDATED


def _judge_email(match: re.Match[str], ctx: _Context) -> _Verdict:
    return PiiType.EMAIL, SCORE_EMAIL


def _judge_passport(match: re.Match[str], ctx: _Context) -> _Verdict:
    window = range(-_PASSPORT_KEYWORD_LINES, _PASSPORT_KEYWORD_LINES + 1)
    if any(_PASSPORT_KEYWORD.search(ctx.text_at(offset)) for offset in window):
        return PiiType.PASSPORT, SCORE_PASSPORT_WITH_KEYWORD
    return PiiType.PASSPORT, SCORE_PASSPORT


def _judge_account(match: re.Match[str], ctx: _Context) -> _Verdict:
    if not 10 <= len(_digits_of(match.group())) <= 14:
        return None
    # 주문번호·송장번호·사업자번호와 모양이 같아서 같은 줄이나 바로 윗줄에 키워드가 있을 때만 받는다
    if not (_ACCOUNT_KEYWORD.search(ctx.line.text) or _ACCOUNT_KEYWORD.search(ctx.text_at(-1))):
        return None
    return PiiType.ACCOUNT, SCORE_ACCOUNT


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
    # 운전면허번호: 지역 코드 검사
    _Rule(_DRIVER_LICENSE, _judge_driver_license),
    # 전화번호: 국번·자릿수 검사
    _Rule(_PHONE, _judge_phone),
    # 여권번호: 형식만 보고 잡는 경우가 많다. 영문+숫자라 숫자 항목과는 거의 겹치지 않는다
    _Rule(_PASSPORT, _judge_passport),
    # 가려진 카드번호: Luhn을 볼 수 없다
    _Rule(_MASKED_CARD, _judge_masked_card),
    # 계좌번호: 검증 수단 없이 키워드로만 잡으므로 맨 뒤에 둔다.
    # 주민번호(6-7), 운전면허번호(2-2-6-2, 구형 2-6-2), 전화번호(`010-1234-5678`)가 모두 계좌 모양이라
    # 계좌 키워드가 있어도 앞 항목으로 본다. 휴대폰 번호를 계좌번호로 쓰는 평생계좌여도 가려지는
    # 범위는 같다. 카드번호는 15·16자리라 계좌(10~14자리)와 길이로 갈린다
    _Rule(_ACCOUNT, _judge_account),
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
        texts = {line.id: line.text for line in lines}
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
            for cand in _resolve(_candidates(_Context(line, texts)))
        ]
