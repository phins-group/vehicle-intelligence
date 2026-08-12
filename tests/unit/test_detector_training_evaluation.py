from __future__ import annotations

import json

from vehicle_intelligence.training.config import DetectorReleaseGates
from vehicle_intelligence.training.evaluation import (
    detector_release_gate_failures,
    evaluate_detector,
    evaluate_detector_files,
)

from .training_fixtures import build_detector_dataset


def _ground_truth() -> dict:
    return {
        "images": [
            {
                "id": 1,
                "group_id": "track-day",
                "camera_id": "gate-a",
                "attributes": {"lighting": "DAY"},
            },
            {
                "id": 2,
                "group_id": "track-night",
                "camera_id": "gate-b",
                "attributes": {"lighting": "NIGHT"},
            },
        ],
        "categories": [{"id": 1, "name": "license_plate"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 20]},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [20, 20, 40, 20]},
        ],
    }


def test_detector_metrics_include_operational_slices_and_gate_failures() -> None:
    evaluation = evaluate_detector(
        _ground_truth(),
        [
            {"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 20], "score": 0.9},
            {"image_id": 2, "category_id": 1, "bbox": [70, 50, 10, 10], "score": 0.8},
        ],
    )
    gates = DetectorReleaseGates(
        minimum_precision=0.9,
        minimum_recall=0.9,
        minimum_group_recall=0.9,
        minimum_night_recall=0.9,
        minimum_full_bbox_coverage=0.9,
        minimum_per_class_recall={"license_plate": 0.9},
    )

    assert evaluation.precision == 0.5
    assert evaluation.recall == 0.5
    assert evaluation.group_recall == 0.5
    assert evaluation.by_lighting["NIGHT"].recall == 0
    assert evaluation.full_bbox_coverage == 0.5
    assert detector_release_gate_failures(evaluation, gates) == [
        "precision",
        "recall",
        "group_recall",
        "full_bbox_coverage",
        "night_recall",
        "class_recall:license_plate",
    ]


def test_perfect_predictions_reach_perfect_ap_and_file_evaluation(tmp_path) -> None:
    dataset, _ = build_detector_dataset(tmp_path)
    annotations_path = dataset / "annotations/test.json"
    annotations = json.loads(annotations_path.read_text())
    predictions = [
        {
            "image_id": item["image_id"],
            "category_id": item["category_id"],
            "bbox": item["bbox"],
            "score": 0.99,
        }
        for item in annotations["annotations"]
    ]
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(json.dumps(predictions))

    evaluation = evaluate_detector_files(annotations_path, predictions_path)

    assert evaluation.map50 == 1
    assert evaluation.map50_95 == 1
    assert evaluation.recall == 1
    assert evaluation.precision == 1
