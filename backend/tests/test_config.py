import pytest
from pydantic import ValidationError

from backend.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    import os

    for key in list(os.environ):
        if key.startswith("GARIGAE_"):
            monkeypatch.delenv(key)


def test_defaults():
    settings = Settings()
    assert settings.ocr_languages == ["ko", "en"]
    assert settings.ocr_device == "auto"
    assert settings.ocr_model_dir is None
    assert settings.ocr_rotations == []
    assert settings.max_upload_bytes == 10 * 1024 * 1024
    assert settings.detectors == ["regex", "rules"]


def test_comma_separated_lists(monkeypatch):
    monkeypatch.setenv("GARIGAE_DETECTORS", "regex, rules,ner")
    monkeypatch.setenv("GARIGAE_OCR_ROTATIONS", "90,180,270")
    monkeypatch.setenv("GARIGAE_CORS_ORIGINS", "https://garigae.example.com")
    settings = Settings()
    assert settings.detectors == ["regex", "rules", "ner"]
    assert settings.ocr_rotations == [90, 180, 270]
    assert settings.cors_origins == ["https://garigae.example.com"]


def test_empty_rotations_disables_rotation(monkeypatch):
    monkeypatch.setenv("GARIGAE_OCR_ROTATIONS", "")
    assert Settings().ocr_rotations == []


@pytest.mark.parametrize(
    ("device", "gpu"),
    [("auto", True), ("cpu", False), ("cuda", "cuda"), ("mps", "mps")],
)
def test_ocr_gpu_argument(monkeypatch, device, gpu):
    monkeypatch.setenv("GARIGAE_OCR_DEVICE", device)
    assert Settings().ocr_gpu == gpu


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("GARIGAE_DETECTORS", "regex,llm"),
        ("GARIGAE_OCR_ROTATIONS", "45"),
        ("GARIGAE_OCR_ROTATIONS", "abc"),
        ("GARIGAE_OCR_DEVICE", "tpu"),
    ],
)
def test_rejects_unknown_values(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        Settings()
