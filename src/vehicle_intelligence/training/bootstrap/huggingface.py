"""Pinned Hugging Face dataset-server adapter for plate bootstrap samples."""

from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import quote

import cv2
import numpy as np

from vehicle_intelligence.exceptions import SampleDataAcquisitionError
from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.http import BootstrapHttpClient
from vehicle_intelligence.training.domain import DetectorSample

_REPO_ID = "justjuu/license-plate-detection"
_REVISION = "b76dbba86154c33fa370bc3087fbc7c766845a66"
_DATASET_URL = f"https://huggingface.co/datasets/{_REPO_ID}"
_INFO_URL = f"https://huggingface.co/api/datasets/{_REPO_ID}"
_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?"
    f"dataset={quote(_REPO_ID, safe='')}&config=default&split=train&offset=0&length={{count}}"
)
_CAPTURED_AT = datetime(2026, 1, 30, 6, 55, 40, tzinfo=UTC)

SOURCE_INFO = BootstrapSourceInfo(
    source_id="justjuu-license-plate-detection",
    dataset_url=_DATASET_URL,
    revision=_REVISION,
    annotation_license="CC-BY-4.0",
    image_license="CC-BY-4.0-DATASET-CARD",
)


class HuggingFacePlateSampleSource:
    def __init__(self, http: BootstrapHttpClient) -> None:
        self._http = http

    def acquire(self, *, sample_count: int) -> list[AcquiredDetectorSample]:
        if not 1 <= sample_count <= 100:
            raise SampleDataAcquisitionError("plate sample count must be between 1 and 100")
        info = self._http.get_json(_INFO_URL, maximum_bytes=5_000_000)
        if info.get("sha") != _REVISION or info.get("private") is not False:
            raise SampleDataAcquisitionError("plate bootstrap repository revision changed")
        payload = self._http.get_json(
            _ROWS_URL.format(count=sample_count),
            maximum_bytes=20_000_000,
        )
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != sample_count:
            raise SampleDataAcquisitionError("plate bootstrap rows response is incomplete")
        return [self._sample_from_row(item) for item in rows]

    def _sample_from_row(self, item: object) -> AcquiredDetectorSample:
        if not isinstance(item, dict) or not isinstance(item.get("row"), dict):
            raise SampleDataAcquisitionError("plate bootstrap row is invalid")
        row = item["row"]
        row_index = item.get("row_idx")
        image_info = row.get("image")
        objects = row.get("objects")
        if (
            not isinstance(row_index, int)
            or not isinstance(image_info, dict)
            or not isinstance(objects, dict)
            or not isinstance(image_info.get("src"), str)
        ):
            raise SampleDataAcquisitionError("plate bootstrap row contract is invalid")
        source_url = image_info["src"]
        if _REVISION not in source_url:
            raise SampleDataAcquisitionError("plate image asset does not match pinned revision")
        data = self._http.get_bytes(source_url, maximum_bytes=20_000_000)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise SampleDataAcquisitionError("plate bootstrap image cannot be decoded")
        height, width = image.shape[:2]
        if image_info.get("width") != width or image_info.get("height") != height:
            raise SampleDataAcquisitionError("plate bootstrap image dimensions changed")
        boxes = objects.get("bbox")
        categories = objects.get("category")
        if (
            not isinstance(boxes, list)
            or not isinstance(categories, list)
            or len(boxes) != len(categories)
            or not boxes
        ):
            raise SampleDataAcquisitionError("plate bootstrap annotations are invalid")
        annotations = []
        for box, category in zip(boxes, categories, strict=True):
            if category != 0 or not isinstance(box, list) or len(box) != 4:
                raise SampleDataAcquisitionError("plate bootstrap class/bbox is invalid")
            try:
                x, y, box_width, box_height = (float(value) for value in box)
            except (TypeError, ValueError) as exc:
                raise SampleDataAcquisitionError("plate bootstrap bbox is not numeric") from exc
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width + 1e-6
                or y + box_height > height + 1e-6
            ):
                raise SampleDataAcquisitionError("plate bootstrap bbox exceeds image")
            annotations.append(
                {
                    "className": "license_plate",
                    "bbox": {
                        "x": x,
                        "y": y,
                        "width": box_width,
                        "height": box_height,
                    },
                    "attributes": {"sourceCategory": 0},
                }
            )
        sample_id = f"hf-plate-{_REVISION[:12]}-{row_index:06d}"
        sample = DetectorSample.model_validate(
            {
                "sampleId": sample_id,
                "imagePath": f"images/huggingface/{row_index:06d}.jpg",
                "groupId": sample_id,
                "cameraId": "external-huggingface",
                "capturedAt": _CAPTURED_AT,
                "attributes": {
                    "sourceDataset": SOURCE_INFO.source_id,
                    "sourceRevision": SOURCE_INFO.revision,
                    "sourceLicense": SOURCE_INFO.image_license,
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
                "license": SOURCE_INFO.image_license,
                "author": "justjuu / upstream Roboflow contributors",
                "landing_url": SOURCE_INFO.dataset_url,
            },
        )
