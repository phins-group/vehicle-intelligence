from __future__ import annotations

import hashlib
import sys
from types import ModuleType

import pytest

from vehicle_intelligence.config import OCRConfig
from vehicle_intelligence.exceptions import ModelLoadError
from vehicle_intelligence.infrastructure.vision import paddleocr
from vehicle_intelligence.model_artifact import sha256_directory


def _model_directory(tmp_path, name: str, payload: bytes):
    directory = tmp_path / name
    directory.mkdir()
    (directory / "inference.json").write_bytes(payload)
    return directory, sha256_directory(directory)


def _fake_paddleocr(monkeypatch):
    calls = []
    module = ModuleType("paddleocr")

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

    module.PaddleOCR = FakePaddleOCR
    monkeypatch.setitem(sys.modules, "paddleocr", module)
    monkeypatch.setattr(paddleocr, "find_spec", lambda _name: object())
    return calls


def test_local_paddle_models_are_verified_and_paired_with_pinned_names(
    tmp_path,
    monkeypatch,
) -> None:
    detection, detection_hash = _model_directory(tmp_path, "detection", b"detector")
    recognition, recognition_hash = _model_directory(tmp_path, "recognition", b"recognizer")
    calls = _fake_paddleocr(monkeypatch)
    config = OCRConfig(
        detection_model_directory=str(detection),
        detection_model_hash=detection_hash,
        recognition_model_directory=str(recognition),
        recognition_model_hash=recognition_hash,
    )

    provider = paddleocr.PaddleOCRProvider(config, require_local_artifacts=True)

    assert calls[0]["text_detection_model_dir"] == str(detection.resolve())
    assert calls[0]["text_recognition_model_dir"] == str(recognition.resolve())
    assert calls[0]["text_detection_model_name"] == config.detection_model_name
    assert calls[0]["text_recognition_model_name"] == config.model_name
    expected_stack_hash = hashlib.sha256(
        f"detection:{detection_hash}\nrecognition:{recognition_hash}\n".encode()
    ).hexdigest()
    assert provider._metadata.hash == expected_stack_hash


def test_paddle_model_hash_mismatch_fails_before_provider_initialization(
    tmp_path,
    monkeypatch,
) -> None:
    detection, _ = _model_directory(tmp_path, "detection", b"detector")
    recognition, recognition_hash = _model_directory(tmp_path, "recognition", b"recognizer")
    calls = _fake_paddleocr(monkeypatch)

    with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
        paddleocr.PaddleOCRProvider(
            OCRConfig(
                detection_model_directory=str(detection),
                detection_model_hash="0" * 64,
                recognition_model_directory=str(recognition),
                recognition_model_hash=recognition_hash,
            ),
            require_local_artifacts=True,
        )

    assert not calls


def test_production_paddle_provider_refuses_managed_downloads(monkeypatch) -> None:
    calls = _fake_paddleocr(monkeypatch)

    with pytest.raises(ModelLoadError, match="production OCR requires local"):
        paddleocr.PaddleOCRProvider(OCRConfig(), require_local_artifacts=True)

    assert not calls


def test_legacy_hash_is_not_reported_as_verified_for_managed_models(monkeypatch) -> None:
    calls = _fake_paddleocr(monkeypatch)

    provider = paddleocr.PaddleOCRProvider(OCRConfig(model_hash="a" * 64))

    assert calls[0]["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert calls[0]["text_recognition_model_name"] == "PP-OCRv5_mobile_rec"
    assert provider._metadata.hash is None
