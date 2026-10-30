"""EasyOCR 모델(`GARIGAE_OCR_LANGUAGES`)을 `GARIGAE_OCR_MODEL_DIR`에 미리 받아 둔다.

대회장 네트워크가 불안할 때와 Docker 이미지 빌드용. GPU 없이도 돈다.

    uv run python scripts/download_models.py
"""

import easyocr

from backend.config import get_settings


def main() -> None:
    settings = get_settings()
    model_dir = str(settings.ocr_model_dir) if settings.ocr_model_dir else None
    reader = easyocr.Reader(
        settings.ocr_languages,
        gpu=False,
        quantize=False,  # 받기만 하므로 CPU 양자화는 생략
        model_storage_directory=model_dir,
        verbose=False,
    )
    print(f"languages: {settings.ocr_languages}")
    print(f"model dir: {reader.model_storage_directory}")


if __name__ == "__main__":
    main()
