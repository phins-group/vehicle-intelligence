from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.training.bootstrap.huggingface import (
    SOURCE_INFO as PLATE_SOURCE_INFO,
)
from vehicle_intelligence.training.bootstrap.huggingface import (
    HuggingFacePlateSampleSource,
)
from vehicle_intelligence.training.bootstrap.open_images import (
    SOURCE_INFO as VEHICLE_SOURCE_INFO,
)
from vehicle_intelligence.training.bootstrap.open_images import (
    OpenImagesVehicleSampleSource,
)
from vehicle_intelligence.training.bootstrap.writer import (
    BootstrapSourceWriter,
    verify_bootstrap_source,
)
from vehicle_intelligence.training.config import DetectorDatasetConfig, SplitConfig
from vehicle_intelligence.training.dataset import (
    DetectorDatasetBuilder,
    verify_detector_dataset,
)
from vehicle_intelligence.training.domain import DetectorRole


class _OpenImagesHttp:
    def get_bytes(self, url: str, *, maximum_bytes: int) -> bytes:
        assert maximum_bytes > 0
        if "class-descriptions" in url:
            return b"/m/car,Car\n/m/motorcycle,Motorcycle\n/m/bus,Bus\n/m/truck,Truck\n"
        if "annotations-bbox" in url:
            return (
                b"ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,"
                b"IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n"
                b"0001,xclick,/m/car,1,0.1,0.8,0.2,0.9,0,0,0,0,0\n"
                b"0002,xclick,/m/motorcycle,1,0.2,0.7,0.1,0.8,0,0,0,0,0\n"
                b"0003,xclick,/m/bus,1,0.1,0.9,0.1,0.9,0,0,0,0,0\n"
                b"0004,xclick,/m/truck,1,0.1,0.9,0.2,0.8,0,0,0,0,0\n"
            )
        if "images-with-rotation" in url:
            rows = [
                "ImageID,License,Author,OriginalLandingURL",
                *[
                    f"000{index},https://creativecommons.org/licenses/by/2.0/,"
                    f"Author {index},https://example.invalid/000{index}"
                    for index in range(1, 5)
                ],
            ]
            return ("\n".join(rows) + "\n").encode()
        assert "open-images-dataset.s3.amazonaws.com" in url
        image_id = int(url.rsplit("/", maxsplit=1)[-1].removesuffix(".jpg"))
        return _jpeg(fill=100 + image_id)

    def get_json(self, url: str, *, maximum_bytes: int) -> dict[str, Any]:
        raise AssertionError(f"unexpected JSON request: {url} {maximum_bytes}")


class _PlateHttp:
    def __init__(self, image: bytes) -> None:
        self._image = image

    def get_bytes(self, url: str, *, maximum_bytes: int) -> bytes:
        assert maximum_bytes > 0
        assert "cached-assets" in url
        return self._image

    def get_json(self, url: str, *, maximum_bytes: int) -> dict[str, Any]:
        assert maximum_bytes > 0
        if "/api/datasets/" in url:
            return {
                "sha": PLATE_SOURCE_INFO.revision,
                "private": False,
            }
        return {
            "rows": [
                {
                    "row_idx": index,
                    "row": {
                        "image": {
                            "src": (
                                "https://datasets-server.huggingface.co/cached-assets/"
                                f"sample/{PLATE_SOURCE_INFO.revision}/{index}/image.jpg"
                            ),
                            "width": 96,
                            "height": 64,
                        },
                        "objects": {
                            "bbox": [[10.0, 20.0, 50.0, 20.0]],
                            "category": [0],
                        },
                    },
                }
                for index in range(2)
            ]
        }


def test_open_images_vehicle_bootstrap_is_attributed_and_not_acceptance_data(
    tmp_path: Path,
) -> None:
    samples = OpenImagesVehicleSampleSource(_OpenImagesHttp()).acquire(samples_per_class=1)
    source = tmp_path / "source" / "vehicle"
    result = BootstrapSourceWriter(DetectorRole.VEHICLE, source).write(
        VEHICLE_SOURCE_INFO,
        samples,
    )
    manifest, digest = verify_bootstrap_source(source)

    assert len(samples) == 4
    assert result.manifest_sha256 == digest
    assert manifest["acceptanceEligible"] is False
    assert manifest["annotationCount"] == 4
    assert len((source / "ATTRIBUTION.csv").read_text().splitlines()) == 5

    config = DetectorDatasetConfig(
        role=DetectorRole.VEHICLE,
        source_directory=source,
        output_directory=tmp_path / "exports",
        classes=("car", "motorcycle", "bus", "truck"),
        split=SplitConfig(require_non_empty=False),
    )
    dataset = DetectorDatasetBuilder(config).build("vehicle-bootstrap-v1").directory
    dataset_manifest, _ = verify_detector_dataset(dataset)
    assert dataset_manifest["acceptanceEligible"] is False
    assert dataset_manifest["licenseStatus"] == "REVIEW_REQUIRED"
    assert dataset_manifest["source"]["id"] == "open-images-v7-validation"
    assert (dataset / "BOOTSTRAP_ONLY.md").is_file()
    assert len((dataset / "ATTRIBUTION.csv").read_text().splitlines()) == 5
    card = (dataset / "README.md").read_text()
    assert "license: other" in card
    assert "BOOTSTRAP ONLY" in card


def test_huggingface_plate_bootstrap_pins_revision_and_canonical_bbox(tmp_path: Path) -> None:
    samples = HuggingFacePlateSampleSource(_PlateHttp(_jpeg())).acquire(sample_count=2)
    source = tmp_path / "source" / "plate"
    result = BootstrapSourceWriter(DetectorRole.PLATE, source).write(
        PLATE_SOURCE_INFO,
        samples,
    )
    manifest, digest = verify_bootstrap_source(source)

    assert result.manifest_sha256 == digest
    assert manifest["sampleCount"] == 2
    assert samples[0].sample.annotations[0].class_name == "license_plate"
    assert samples[0].sample.annotations[0].bbox.as_xywh() == (10.0, 20.0, 50.0, 20.0)


def _jpeg(*, fill: int = 127) -> bytes:
    image = np.full((64, 96, 3), fill, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
