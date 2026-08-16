import hashlib

import numpy as np
import pytest

from vehicle_intelligence.config import VehicleEmbeddingConfig
from vehicle_intelligence.exceptions import ModelLoadError

torch = pytest.importorskip("torch", reason="vehicle embedding tests require the reid extra")
vehicle_embedding = pytest.importorskip(
    "vehicle_intelligence.infrastructure.vision.vehicle_embedding",
    reason="vehicle embedding tests require the reid extra",
)
TorchScriptVehicleEmbeddingProvider = vehicle_embedding.TorchScriptVehicleEmbeddingProvider


class TinyEmbedding(torch.nn.Module):
    def forward(self, image):
        return image.mean(dim=(2, 3))


def test_torchscript_embedding_is_versioned_hashed_and_normalized(tmp_path) -> None:
    path = tmp_path / "tiny-reid.pt"
    traced = torch.jit.trace(TinyEmbedding(), torch.zeros((1, 3, 32, 32)))
    torch.jit.save(traced, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    provider = TorchScriptVehicleEmbeddingProvider(
        VehicleEmbeddingConfig(
            enabled=True,
            model_path=path,
            model_name="tiny-reid",
            model_version="test-1",
            model_hash=digest,
            dimension=3,
            image_width=32,
            image_height=32,
        )
    )

    vector = provider.embed(np.full((40, 60, 3), 128, dtype=np.uint8))

    assert len(vector) == 3
    assert np.linalg.norm(vector) == pytest.approx(1)
    assert provider.model.version == "test-1"
    assert provider.model.model_hash == digest


def test_torchscript_embedding_rejects_checkpoint_hash_mismatch(tmp_path) -> None:
    path = tmp_path / "tiny-reid.pt"
    torch.jit.save(
        torch.jit.trace(TinyEmbedding(), torch.zeros((1, 3, 32, 32))),
        path,
    )
    with pytest.raises(ModelLoadError, match="hash"):
        TorchScriptVehicleEmbeddingProvider(
            VehicleEmbeddingConfig(
                enabled=True,
                model_path=path,
                model_hash="0" * 64,
                dimension=3,
                image_width=32,
                image_height=32,
            )
        )
