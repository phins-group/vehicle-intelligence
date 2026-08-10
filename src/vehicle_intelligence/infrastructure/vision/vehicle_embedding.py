"""Versioned TorchScript vehicle embedding adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.config import VehicleEmbeddingConfig
from vehicle_intelligence.domain import EmbeddingModel
from vehicle_intelligence.exceptions import (
    DependencyUnavailableError,
    InferenceError,
    ModelLoadError,
)


class TorchScriptVehicleEmbeddingProvider:
    def __init__(self, config: VehicleEmbeddingConfig) -> None:
        if not config.enabled or config.model_path is None:
            raise ModelLoadError("vehicle embedding model is not enabled/configured")
        self._config = config
        self._path = Path(config.model_path)
        if not self._path.is_file():
            raise ModelLoadError(f"vehicle embedding checkpoint not found: {self._path}")
        digest = _sha256(self._path)
        if config.model_hash is not None and digest.casefold() != config.model_hash.casefold():
            raise ModelLoadError("vehicle embedding checkpoint hash does not match config")
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError(
                "TorchScript vehicle embeddings require the 'reid' dependency extra"
            ) from exc
        try:
            self._torch = torch
            self._device = torch.device(config.device)
            self._network: Any = torch.jit.load(str(self._path), map_location=self._device)
            self._network.eval()
        except Exception as exc:
            raise ModelLoadError("cannot load TorchScript vehicle embedding model") from exc
        self._model = EmbeddingModel(
            name=config.model_name,
            version=config.model_version,
            model_hash=digest,
            dimension=config.dimension,
        )

    @property
    def model(self) -> EmbeddingModel:
        return self._model

    def embed(self, image: NDArray[np.uint8]) -> tuple[float, ...]:
        if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
            raise InferenceError("vehicle embedding input must be a non-empty BGR image")
        try:
            resized = cv2.resize(
                image,
                (self._config.image_width, self._config.image_height),
                interpolation=cv2.INTER_AREA,
            )
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            array = rgb.astype(np.float32) / 255.0
            array = (array - np.asarray((0.485, 0.456, 0.406), dtype=np.float32)) / np.asarray(
                (0.229, 0.224, 0.225), dtype=np.float32
            )
            tensor = self._torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
            tensor = tensor.to(self._device)
            with self._torch.inference_mode():
                output = self._network(tensor)
            if not self._torch.is_tensor(output):
                raise TypeError("embedding model output is not a tensor")
            vector = output.detach().float().cpu().reshape(-1).numpy()
            if vector.size != self._model.dimension or not np.isfinite(vector).all():
                raise ValueError("embedding output dimension/values are invalid")
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                raise ValueError("embedding output has zero norm")
            return tuple(float(value) for value in vector / norm)
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError("vehicle embedding inference failed") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelLoadError(f"cannot read vehicle embedding checkpoint: {path}") from exc
    return digest.hexdigest()
