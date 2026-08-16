"""Small, class-balanced Open Images vehicle sample adapter."""

from __future__ import annotations

import csv
import io
from collections import Counter
from datetime import UTC, datetime
from itertools import groupby

import cv2
import numpy as np

from vehicle_intelligence.exceptions import SampleDataAcquisitionError
from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.http import BootstrapHttpClient
from vehicle_intelligence.training.domain import DetectorSample

_DATASET_PAGE = "https://storage.googleapis.com/openimages/web/download_v7.html"
_CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
_BOX_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
_METADATA_URL = (
    "https://storage.googleapis.com/openimages/2018_04/validation/"
    "validation-images-with-rotation.csv"
)
_IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
_TARGET_CLASSES = ("car", "motorcycle", "bus", "truck")
_CAPTURED_AT = datetime(2022, 10, 1, tzinfo=UTC)

SOURCE_INFO = BootstrapSourceInfo(
    source_id="open-images-v7-validation",
    dataset_url=_DATASET_PAGE,
    revision="v7-static-2022-10",
    annotation_license="CC-BY-4.0",
    image_license="PER_IMAGE_CC-BY-2.0_LISTED",
)


class OpenImagesVehicleSampleSource:
    def __init__(self, http: BootstrapHttpClient) -> None:
        self._http = http

    def acquire(self, *, samples_per_class: int) -> list[AcquiredDetectorSample]:
        if not 1 <= samples_per_class <= 100:
            raise SampleDataAcquisitionError("vehicle sample count must be between 1 and 100")
        mids = self._target_mids()
        boxes = self._select_boxes(mids, samples_per_class)
        metadata = self._image_metadata(set(boxes))
        samples = [
            self._download_sample(image_id, rows, mids, metadata[image_id])
            for image_id, rows in sorted(boxes.items())
        ]
        if not samples:
            raise SampleDataAcquisitionError("Open Images selection produced no samples")
        return samples

    def _target_mids(self) -> dict[str, str]:
        raw = self._http.get_bytes(_CLASS_URL, maximum_bytes=5_000_000)
        rows = csv.reader(io.StringIO(_decode_csv(raw)))
        by_name = {name.strip().lower(): mid.strip() for mid, name in rows}
        missing = [name for name in _TARGET_CLASSES if name not in by_name]
        if missing:
            raise SampleDataAcquisitionError(
                f"Open Images class mapping is missing: {', '.join(missing)}"
            )
        return {by_name[name]: name for name in _TARGET_CLASSES}

    def _select_boxes(
        self,
        mids: dict[str, str],
        samples_per_class: int,
    ) -> dict[str, list[dict[str, str]]]:
        raw = self._http.get_bytes(_BOX_URL, maximum_bytes=250_000_000)
        reader = csv.DictReader(io.StringIO(_decode_csv(raw)))
        selected: dict[str, list[dict[str, str]]] = {}
        counts: Counter[str] = Counter()
        for image_id, group in groupby(reader, key=lambda row: row.get("ImageID", "")):
            relevant = [row for row in group if _usable_box(row, mids)]
            present = {mids[row["LabelName"]] for row in relevant}
            if not present or not any(counts[name] < samples_per_class for name in present):
                continue
            selected[image_id] = relevant
            counts.update(present)
            if all(counts[name] >= samples_per_class for name in _TARGET_CLASSES):
                break
        missing = [name for name in _TARGET_CLASSES if counts[name] < samples_per_class]
        if missing:
            raise SampleDataAcquisitionError(
                f"Open Images cannot satisfy class-balanced sample: {', '.join(missing)}"
            )
        return selected

    def _image_metadata(self, image_ids: set[str]) -> dict[str, dict[str, str]]:
        raw = self._http.get_bytes(_METADATA_URL, maximum_bytes=100_000_000)
        reader = csv.DictReader(io.StringIO(_decode_csv(raw)))
        selected = {row["ImageID"]: row for row in reader if row.get("ImageID") in image_ids}
        missing = sorted(image_ids - selected.keys())
        if missing:
            raise SampleDataAcquisitionError("Open Images attribution metadata is incomplete")
        for row in selected.values():
            if "creativecommons.org/licenses/by/" not in row.get("License", ""):
                raise SampleDataAcquisitionError(
                    "Open Images sample does not declare an attribution license"
                )
        return selected

    def _download_sample(
        self,
        image_id: str,
        rows: list[dict[str, str]],
        mids: dict[str, str],
        metadata: dict[str, str],
    ) -> AcquiredDetectorSample:
        image_url = _IMAGE_URL.format(image_id=image_id)
        data = self._http.get_bytes(image_url, maximum_bytes=20_000_000)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise SampleDataAcquisitionError("Open Images sample cannot be decoded")
        height, width = image.shape[:2]
        annotations = []
        for row in rows:
            x1 = _coordinate(row, "XMin") * width
            x2 = _coordinate(row, "XMax") * width
            y1 = _coordinate(row, "YMin") * height
            y2 = _coordinate(row, "YMax") * height
            x1, x2 = max(0.0, x1), min(float(width), x2)
            y1, y2 = max(0.0, y1), min(float(height), y2)
            if x2 <= x1 or y2 <= y1:
                raise SampleDataAcquisitionError("Open Images bounding box is invalid")
            annotations.append(
                {
                    "className": mids[row["LabelName"]],
                    "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                    "attributes": {
                        "source": row.get("Source"),
                        "occluded": row.get("IsOccluded") == "1",
                        "truncated": row.get("IsTruncated") == "1",
                    },
                }
            )
        sample_id = f"open-images-v7-validation-{image_id}"
        sample = DetectorSample.model_validate(
            {
                "sampleId": sample_id,
                "imagePath": f"images/open-images/{image_id}.jpg",
                "groupId": sample_id,
                "cameraId": "external-open-images",
                "capturedAt": _CAPTURED_AT,
                "attributes": {
                    "sourceDataset": SOURCE_INFO.source_id,
                    "sourceRevision": SOURCE_INFO.revision,
                    "sourceLicense": metadata["License"],
                    "sourceAuthor": metadata.get("Author") or "UNKNOWN",
                    "sourceLandingUrl": metadata.get("OriginalLandingURL") or "UNKNOWN",
                    "licenseReviewStatus": "REVIEW_REQUIRED",
                    "acceptanceEligible": False,
                    "bootstrapOnly": True,
                },
                "annotations": annotations,
            }
        )
        return AcquiredDetectorSample(
            sample=sample,
            image_bytes=data,
            attribution={
                "sample_id": sample_id,
                "source_dataset": SOURCE_INFO.source_id,
                "source_revision": SOURCE_INFO.revision,
                "license": metadata["License"],
                "author": metadata.get("Author") or "UNKNOWN",
                "landing_url": metadata.get("OriginalLandingURL") or "",
            },
        )


def _usable_box(row: dict[str, str], mids: dict[str, str]) -> bool:
    return (
        row.get("LabelName") in mids
        and row.get("IsGroupOf") == "0"
        and row.get("IsDepiction") == "0"
        and row.get("IsInside") == "0"
    )


def _coordinate(row: dict[str, str], name: str) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise SampleDataAcquisitionError("Open Images coordinate is invalid") from exc
    if not 0 <= value <= 1:
        raise SampleDataAcquisitionError("Open Images coordinate is outside [0, 1]")
    return value


def _decode_csv(raw: bytes) -> str:
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise SampleDataAcquisitionError("bootstrap CSV is not UTF-8") from exc
