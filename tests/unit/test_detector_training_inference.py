from __future__ import annotations

import json

from vehicle_intelligence.domain import BoundingBox, Detection, ModelMetadata
from vehicle_intelligence.training.domain import DatasetSplit, DetectorRole
from vehicle_intelligence.training.evaluation import evaluate_detector_files
from vehicle_intelligence.training.inference import predict_dataset_split

from .training_fixtures import build_detector_dataset


class _TruckDetector:
    def detect(self, image):
        assert image.shape == (80, 120, 3)
        return [
            Detection(
                bbox=BoundingBox(10, 15, 70, 45),
                confidence=0.99,
                class_id=3,
                class_name="truck",
                model=ModelMetadata("vehicle-test", "v1", "a" * 64),
            )
        ]


def test_canonical_detector_creates_checksum_traced_coco_predictions(tmp_path) -> None:
    dataset, _ = build_detector_dataset(tmp_path)
    output = tmp_path / "predictions.json"

    result = predict_dataset_split(
        dataset,
        DatasetSplit.TEST,
        DetectorRole.VEHICLE,
        _TruckDetector(),
        output,
        model_info={"name": "vehicle-test", "version": "v1"},
    )
    prediction_manifest = json.loads(result.manifest_path.read_text())
    evaluation = evaluate_detector_files(dataset / "annotations/test.json", output)

    assert result.image_count == 1
    assert result.prediction_count == 1
    assert prediction_manifest["predictionsSha256"] == result.predictions_sha256
    assert prediction_manifest["model"]["name"] == "vehicle-test"
    assert evaluation.precision == 1
    assert evaluation.recall == 1
