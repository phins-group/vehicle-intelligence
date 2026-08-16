"""Local PaddleOCR v3 adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import OCRConfig
from vehicle_intelligence.domain import ModelMetadata, OCRResult
from vehicle_intelligence.exceptions import (
    DependencyUnavailableError,
    InferenceError,
    ModelLoadError,
)
from vehicle_intelligence.model_artifact import validated_model_directory


class PaddleOCRProvider:
    def __init__(
        self,
        config: OCRConfig,
        *,
        require_local_artifacts: bool = False,
    ) -> None:
        required_values = (
            config.detection_model_directory,
            config.detection_model_hash,
            config.recognition_model_directory,
            config.recognition_model_hash,
        )
        if require_local_artifacts and not all(required_values):
            raise ModelLoadError(
                "production OCR requires local detection/recognition model directories "
                "and SHA-256 manifest hashes"
            )
        model_arguments: dict[str, str] = {
            "text_detection_model_name": config.detection_model_name,
            "text_recognition_model_name": config.model_name,
        }
        verified_hashes: list[tuple[str, str]] = []
        if config.detection_model_directory:
            path, actual_hash = validated_model_directory(
                config.detection_model_directory,
                config.detection_model_hash,
            )
            model_arguments["text_detection_model_dir"] = str(path)
            verified_hashes.append(("detection", actual_hash))
        if config.recognition_model_directory:
            path, actual_hash = validated_model_directory(
                config.recognition_model_directory,
                config.recognition_model_hash,
            )
            model_arguments["text_recognition_model_dir"] = str(path)
            verified_hashes.append(("recognition", actual_hash))
        if find_spec("paddle") is None:
            raise DependencyUnavailableError(
                "PaddleOCR local inference requires the PaddlePaddle engine; "
                "follow the platform-specific installation command in README.md"
            )
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise DependencyUnavailableError(
                "PaddleOCR is not installed; install a compatible local inference engine"
            ) from exc
        verified_stack_hash = (
            hashlib.sha256(
                "".join(f"{role}:{digest}\n" for role, digest in verified_hashes).encode()
            ).hexdigest()
            if verified_hashes
            else None
        )
        self._metadata = ModelMetadata(
            name=config.model_name,
            version=config.model_version,
            hash=verified_stack_hash,
        )
        try:
            self._ocr = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=config.device,
                **model_arguments,
            )
        except Exception as exc:
            raise ModelLoadError(f"cannot initialize PaddleOCR {config.model_name}") from exc

    def recognize(self, image: NDArray[np.uint8]) -> OCRResult:
        try:
            results = list(self._ocr.predict(image))
        except Exception as exc:
            raise InferenceError("PaddleOCR inference failed") from exc
        texts: list[str] = []
        scores: list[float] = []
        for result in results:
            payload = self._payload(result)
            result_texts = payload.get("rec_texts", [])
            result_scores = payload.get("rec_scores", [])
            for text, score in zip(result_texts, result_scores, strict=False):
                compact = str(text).strip()
                if compact:
                    texts.append(compact)
                    scores.append(float(score))
        if not texts:
            return OCRResult(text="", confidence=0.0, model=self._metadata)
        joined = "".join(texts)
        total_characters = sum(len(text) for text in texts)
        confidence = sum(
            score * len(text) for text, score in zip(texts, scores, strict=True)
        ) / max(total_characters, 1)
        return OCRResult(
            text=joined,
            confidence=min(max(float(confidence), 0.0), 1.0),
            model=self._metadata,
        )

    @staticmethod
    def _payload(result: Any) -> Mapping[str, Any]:
        data = getattr(result, "json", {})
        if callable(data):
            data = data()
        if isinstance(data, Mapping) and isinstance(data.get("res"), Mapping):
            return data["res"]
        return data if isinstance(data, Mapping) else {}
