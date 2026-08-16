"""Model artifact integrity helpers shared by application gates and providers."""

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


def sha256_directory(path: Path) -> str:
    """Hash a model directory as a deterministic manifest of path/file digests."""
    digest = hashlib.sha256()
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if candidate.is_symlink():
            raise ModelLoadError(f"model directory contains a symbolic link: {candidate}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ModelLoadError(f"model directory contains a non-file entry: {candidate}")
        files.append(candidate)
    if not files:
        raise ModelLoadError(f"model directory is empty: {path}")
    for candidate in sorted(files, key=lambda item: item.relative_to(path).as_posix()):
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def validated_model_artifact(path_value: str | None, expected_hash: str | None) -> tuple[Path, str]:
    if not path_value:
        raise ModelLoadError("model_path is required for model inference")
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise ModelLoadError(f"model artifact does not exist: {path}")
    actual_hash = sha256_file(path)
    if expected_hash is not None and not hmac.compare_digest(
        expected_hash.lower().removeprefix("sha256:"), actual_hash
    ):
        raise ModelLoadError(f"model artifact SHA-256 mismatch: {path}")
    return path, actual_hash


def validated_model_directory(
    path_value: str | None,
    expected_hash: str | None,
) -> tuple[Path, str]:
    if not path_value:
        raise ModelLoadError("model directory is required for model inference")
    unresolved = Path(path_value).expanduser()
    if unresolved.is_symlink():
        raise ModelLoadError(f"model directory cannot be a symbolic link: {unresolved}")
    path = unresolved.resolve()
    if not path.is_dir():
        raise ModelLoadError(f"model directory does not exist: {path}")
    actual_hash = sha256_directory(path)
    if expected_hash is not None and not hmac.compare_digest(
        expected_hash.lower().removeprefix("sha256:"), actual_hash
    ):
        raise ModelLoadError(f"model directory SHA-256 mismatch: {path}")
    return path, actual_hash
