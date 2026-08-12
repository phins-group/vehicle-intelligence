from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from vehicle_intelligence.training.config import DetectorDatasetConfig
from vehicle_intelligence.training.dataset import DetectorDatasetBuilder
from vehicle_intelligence.training.domain import DetectorRole


def build_detector_dataset(
    tmp_path: Path,
    role: DetectorRole = DetectorRole.VEHICLE,
    *,
    acceptance_eligible: bool = True,
) -> tuple[Path, DetectorDatasetConfig]:
    source = tmp_path / f"source-{role.value}"
    images = source / "images"
    images.mkdir(parents=True)
    classes = (
        ("car", "motorcycle", "bus", "truck")
        if role is DetectorRole.VEHICLE
        else ("license_plate",)
    )
    records = []
    definitions = (
        ("sample-train-a", "group-train", "train", 50, "DAY"),
        ("sample-train-b", "group-train", "train", 70, "NIGHT"),
        ("sample-validation", "group-validation", "validation", 90, "DAY"),
        ("sample-test", "group-test", "test", 110, "NIGHT"),
    )
    for index, (sample_id, group_id, split, fill, lighting) in enumerate(definitions):
        filename = f"images/{sample_id}.jpg"
        image = np.full((80, 120, 3), fill, dtype=np.uint8)
        image[5 + index, 5 + index] = (fill + 1, fill + 2, fill + 3)
        assert cv2.imwrite(str(source / filename), image)
        class_name = classes[index % len(classes)] if role is DetectorRole.VEHICLE else classes[0]
        records.append(
            {
                "sampleId": sample_id,
                "imagePath": filename,
                "groupId": group_id,
                "cameraId": "gate-01" if split != "test" else "gate-02",
                "capturedAt": f"2026-08-{10 + index:02d}T00:00:00Z",
                "split": split,
                "attributes": {
                    "lighting": lighting,
                    "acceptanceEligible": acceptance_eligible,
                },
                "annotations": [
                    {
                        "className": class_name,
                        "bbox": {"x": 10, "y": 15, "width": 60, "height": 30},
                        "attributes": {
                            "layout": "TWO_LINE" if role is DetectorRole.PLATE else None
                        },
                    }
                ],
            }
        )
    (source / "annotations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    config = DetectorDatasetConfig(
        role=role,
        source_directory=source,
        output_directory=tmp_path / "exports",
        classes=classes,
    )
    result = DetectorDatasetBuilder(config).build(f"{role.value}-v1")
    return result.directory, config
