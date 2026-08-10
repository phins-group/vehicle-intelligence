"""Local PaddleOCR v3 adapter."""

from __future__ import annotations

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


class PaddleOCRProvider:
    def __init__(self, config: OCRConfig) -> None:
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
        self._metadata = ModelMetadata(
            name=config.model_name,
            version=config.model_version,
            hash=config.model_hash,
        )
        try:
            self._ocr = PaddleOCR(
                text_detection_model_name=config.detection_model_name,
                text_recognition_model_name=config.model_name,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                device=config.device,
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
