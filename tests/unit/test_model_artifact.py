from __future__ import annotations

import pytest

from vehicle_intelligence.exceptions import ModelLoadError
from vehicle_intelligence.infrastructure.vision.model_artifact import (
    sha256_directory,
    validated_model_directory,
)


def test_model_directory_digest_is_stable_and_detects_tampering(tmp_path) -> None:
    model = tmp_path / "model"
    (model / "inference").mkdir(parents=True)
    (model / "inference" / "model.pdmodel").write_bytes(b"program")
    (model / "inference" / "model.pdiparams").write_bytes(b"parameters")
    expected = sha256_directory(model)

    resolved, actual = validated_model_directory(str(model), f"sha256:{expected}")

    assert resolved == model.resolve()
    assert actual == expected
    (model / "inference" / "model.pdiparams").write_bytes(b"tampered")
    with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
        validated_model_directory(str(model), expected)


def test_model_directory_rejects_empty_and_symlinked_content(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ModelLoadError, match="empty"):
        validated_model_directory(str(empty), None)

    model = tmp_path / "model"
    model.mkdir()
    external = tmp_path / "external.pdmodel"
    external.write_bytes(b"external")
    (model / "model.pdmodel").symlink_to(external)
    with pytest.raises(ModelLoadError, match="symbolic link"):
        validated_model_directory(str(model), None)
