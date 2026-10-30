"""가짜 데이터로 샘플 이미지와 정답 영역(JSON)을 만든다. 실제 개인정보는 쓰지 않는다.

    uv run python scripts/make_samples.py            # tests/fixtures/ 에 저장
    uv run python scripts/make_samples.py --out DIR

정답 JSON (`<이름>.json`):

    {"image": "delivery_label.png", "description": "...", "width": 900, "height": 560,
     "regions": [{"type": "phone", "text": "010-1234-5678", "polygon": [[x, y], ...]}]}

- `polygon`은 그 문자열을 그린 글자 영역(잉크)을 감싸는 사각형이다. 꼭짓점 4개, 시계 방향
- 기울어진 이미지는 똑같이 돌린 좌표다
- `regions`에는 이미지 안의 민감정보를 종류와 상관없이 전부 넣는다 (아직 감지기가 없는 종류 포함)

폰트에 따라 EasyOCR가 읽는 결과가 달라진다. 저장된 샘플은 macOS AppleSDGothicNeo로 만들었다.
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


@dataclass
class Line:
    text: str
    # (문자열, 종류). 문자열은 text 안에 있어야 한다
    pii: list[tuple[str, str]] = field(default_factory=list)
    size: int = 34
    # 말풍선 배경색 (메신저 캡처용)
    bubble: str | None = None


@dataclass
class Sample:
    name: str
    description: str
    lines: list[Line]
    width: int = 900
    background: str = "white"
    image_format: str = "PNG"
    # 기울기 (°, 반시계 방향)
    angle: float = 0.0
    line_gap: int = 30
    margin: int = 50


# 가짜 데이터. 번호는 실제로 쓰이지 않는 형태로 만들었다
DELIVERY = [
    Line("택배 운송장", size=44),
    Line("운송장번호 1234-5678-9012"),
    Line("받는 분: 김가상", pii=[("김가상", "name")]),
    Line("연락처: 010-1234-5678", pii=[("010-1234-5678", "phone")]),
    Line("주소: 부산광역시 해운대구 가상로 34", pii=[("부산광역시 해운대구 가상로 34", "address")]),
    Line(
        "보내는 분: 이테스트 010-9876-5432",
        pii=[("이테스트", "name"), ("010-9876-5432", "phone")],
    ),
]

CONTACT = [
    Line("연락처 안내", size=42),
    Line("담당자: 박가상", pii=[("박가상", "name")]),
    Line("휴대폰: 010-2468-1357", pii=[("010-2468-1357", "phone")]),
    Line("이메일: test@example.com", pii=[("test@example.com", "email")]),
    Line("문의 시간 평일 09:00-18:00"),
]

SAMPLES = [
    Sample("delivery_label", "택배 송장. 라벨 뒤 번호, 이름 뒤 번호", DELIVERY),
    Sample(
        "messenger",
        "메신저 캡처. 말풍선 안 공백으로 띄운 번호",
        # 노란 말풍선에 34px로 그리면 EasyOCR가 첫 0을 영문 O로 읽어서(`O10 2345 6789`) 38px로 그린다
        [
            Line("엄마", size=30),
            Line("내 번호 바뀌었어", size=38, bubble="#fee500"),
            Line("010 2345 6789", pii=[("010 2345 6789", "phone")], size=38, bubble="#fee500"),
            Line("저장해 둬", size=38, bubble="#fee500"),
            Line("알겠어 고마워", size=38, bubble="#e9e9eb"),
        ],
        width=700,
        background="#b2c7d9",
        line_gap=26,
    ),
    Sample(
        "receipt",
        "영수증 (JPEG). 전화번호 아닌 숫자 묶음이 섞여 있음",
        [
            Line("가상마트 테스트점", size=40),
            Line("사업자번호 123-45-67890"),
            Line("주문번호 2026-1030-0001"),
            Line("회원 연락처 010-1111-2222", pii=[("010-1111-2222", "phone")]),
            Line("합계 32,000원"),
        ],
        width=760,
        image_format="JPEG",
    ),
    Sample("tilted_8", "8° 기울어진 연락처 안내", CONTACT, width=640, angle=8),
    Sample("tilted_20", "20° 기울어진 연락처 안내", CONTACT, width=640, angle=20),
]


def find_font() -> str:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return path
    raise SystemExit(f"한글 폰트가 없음: {FONT_CANDIDATES}")


def render(sample: Sample, font_path: str) -> tuple[Image.Image, list[dict]]:
    fonts = {line.size: ImageFont.truetype(font_path, line.size) for line in sample.lines}
    ascent = {size: font.getmetrics()[0] for size, font in fonts.items()}
    descent = {size: font.getmetrics()[1] for size, font in fonts.items()}
    bubble_pad = 14
    height = sample.margin * 2 + sum(
        ascent[line.size] + descent[line.size] + sample.line_gap for line in sample.lines
    )
    image = Image.new("RGB", (sample.width, height), sample.background)
    draw = ImageDraw.Draw(image)

    regions = []
    y = sample.margin
    for line in sample.lines:
        font = fonts[line.size]
        x = sample.margin + (bubble_pad if line.bubble else 0)
        if line.bubble:
            left, top, right, bottom = draw.textbbox((x, y), line.text, font=font)
            draw.rounded_rectangle(
                (left - bubble_pad, top - bubble_pad, right + bubble_pad, bottom + bubble_pad),
                radius=16,
                fill=line.bubble,
            )
        draw.text((x, y), line.text, fill="#111111", font=font)
        for text, pii_type in line.pii:
            start = line.text.index(text)
            offset = font.getlength(line.text[:start])
            left, top, right, bottom = draw.textbbox((x + offset, y), text, font=font)
            regions.append(
                {
                    "type": pii_type,
                    "text": text,
                    "polygon": [[left, top], [right, top], [right, bottom], [left, bottom]],
                }
            )
        y += ascent[line.size] + descent[line.size] + sample.line_gap
    return image, regions


def rotate(image: Image.Image, regions: list[dict], angle: float) -> tuple[Image.Image, list[dict]]:
    """반시계 방향으로 돌린다. 잘리지 않게 캔버스를 넓히고 정답 좌표도 같이 돌린다."""
    width, height = image.size
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    new_width = int(np.ceil(width * cos + height * sin))
    new_height = int(np.ceil(width * sin + height * cos))
    matrix[0, 2] += new_width / 2 - width / 2
    matrix[1, 2] += new_height / 2 - height / 2
    background = tuple(int(c) for c in np.asarray(image)[0, 0])
    rotated = cv2.warpAffine(
        np.asarray(image),
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    moved = []
    for region in regions:
        points = np.asarray(region["polygon"], dtype=np.float64)
        points = np.column_stack([points, np.ones(len(points))]) @ matrix.T
        moved.append({**region, "polygon": points.tolist()})
    return Image.fromarray(rotated), moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    font_path = find_font()

    for sample in SAMPLES:
        image, regions = render(sample, font_path)
        if sample.angle:
            image, regions = rotate(image, regions, sample.angle)
        suffix = {"PNG": "png", "JPEG": "jpg"}[sample.image_format]
        image_name = f"{sample.name}.{suffix}"
        save_options = {"quality": 92} if sample.image_format == "JPEG" else {"optimize": True}
        image.save(args.out / image_name, format=sample.image_format, **save_options)
        answer = {
            "image": image_name,
            "description": sample.description,
            "width": image.width,
            "height": image.height,
            "regions": [
                {**region, "polygon": [[round(x, 1), round(y, 1)] for x, y in region["polygon"]]}
                for region in regions
            ],
        }
        (args.out / f"{sample.name}.json").write_text(
            json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{image_name}: {image.width}x{image.height}, 정답 영역 {len(regions)}개")


if __name__ == "__main__":
    main()
