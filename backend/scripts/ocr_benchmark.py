"""이미지 한 장으로 EasyOCR 장치와 처리 시간을 잰다. 설정은 `GARIGAE_*`를 따른다.

    uv run python scripts/ocr_benchmark.py IMAGE [--runs 3]

가짜 데이터 샘플에만 쓴다. 인식한 글자를 그대로 출력한다.
"""

import argparse
import time

import easyocr

from backend.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    settings = get_settings()
    model_dir = str(settings.ocr_model_dir) if settings.ocr_model_dir else None

    started = time.perf_counter()
    reader = easyocr.Reader(
        settings.ocr_languages,
        gpu=settings.ocr_gpu,
        model_storage_directory=model_dir,
        download_enabled=False,
        verbose=False,
    )
    load_s = time.perf_counter() - started
    rotation_info = settings.ocr_rotations or None

    timings = []
    for _ in range(args.runs + 1):
        started = time.perf_counter()
        results = reader.readtext(args.image, rotation_info=rotation_info)
        timings.append(time.perf_counter() - started)

    print(f"device: {reader.device}, rotations: {settings.ocr_rotations}")
    print(f"model load: {load_s:.2f}s, first run: {timings[0]:.2f}s")
    steady = timings[1:]
    if steady:
        print(f"next {len(steady)} runs avg: {sum(steady) / len(steady):.2f}s")
    print(f"boxes: {len(results)}")
    for box, text, conf in results:
        print(f"  {conf:.2f}  {text}  {[[int(x), int(y)] for x, y in box]}")


if __name__ == "__main__":
    main()
