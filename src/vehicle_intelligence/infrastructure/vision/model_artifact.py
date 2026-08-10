"""Model artifact integrity helpers shared by optimized providers and tooling."""

from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

from vehicle_intelligence.exceptions import ModelLoadError


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validated_model_artifact(path_value: str | None, expected_hash: str | None) -> tuple[Path, str]:
    if not path_value:
        raise ModelLoadError("model_path is required for optimized inference")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ModelLoadError(f"model artifact does not exist: {path}")
    actual_hash = sha256_file(path)
    if expected_hash is not None and not hmac.compare_digest(
        expected_hash.lower().removeprefix("sha256:"), actual_hash
    ):
        raise ModelLoadError(f"model artifact SHA-256 mismatch: {path}")
    return path, actual_hash
