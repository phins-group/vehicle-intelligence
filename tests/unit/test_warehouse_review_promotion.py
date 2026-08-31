from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.domain.dataset_review import (
    DetectorReviewAction,
    DetectorReviewAnnotation,
    DetectorReviewBox,
    DetectorReviewDecision,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.domain import DetectorSample
from vehicle_intelligence.training.first_party import (
    FirstPartyPlateSourceBuilder,
    verify_first_party_detector_source,
)
from vehicle_intelligence.training.warehouse_plate_review import (
    WarehousePlateReviewSourceBuilder,
)
from vehicle_intelligence.training.warehouse_review_promotion import (
    AttestedWarehouseReviewPromotionBuilder,
)


def test_attested_warehouse_promotion_merges_reviewed_samples_and_rights(
    tmp_path: Path,
) -> None:
    base = _base_source(tmp_path)
    review = _review_source(tmp_path)
    decisions = _decisions(review)
    base_before = (base / "source-manifest.json").read_bytes()
    review_before = (review / "source-manifest.json").read_bytes()

    with pytest.raises(DetectorDatasetError, match="requires all review decisions"):
        AttestedWarehouseReviewPromotionBuilder(
            base_source_directory=base,
            review_source_directory=review,
            output_directory=tmp_path / "sources/incomplete-v2",
            target_source_id="incomplete-v2",
            decisions=dict(list(decisions.items())[:1]),
            rights_holder="duyhuynh",
            attested_by="duyhuynh",
        ).build()

    target = tmp_path / "sources/production-v2"
    result = AttestedWarehouseReviewPromotionBuilder(
        base_source_directory=base,
        review_source_directory=review,
        output_directory=target,
        target_source_id="production-v2",
        decisions=decisions,
        rights_holder="duyhuynh",
        attested_by="duyhuynh",
        clock=lambda: datetime(2026, 8, 27, 8, 0, tzinfo=UTC),
    ).build()

    assert result.sample_count == 3
    assert result.annotation_count == 2
    assert result.negative_sample_count == 1
    assert result.promoted_review_count == 2
    assert result.promoted_positive_count == 1
    assert result.promoted_negative_count == 1
    assert result.rejected_count == 0
    manifest, digest = verify_first_party_detector_source(target)
    assert digest == result.manifest_sha256
    assert manifest["parentSource"]["id"] == "production-v1"
    assert manifest["rightsAssertion"] == ("USER_CONFIRMED_FIRST_PARTY_WAREHOUSE_CAMERA_COLLECTION")
    assert manifest["rightsAttestation"]["rightsHolder"] == "duyhuynh"
    assert manifest["warehouseReviewPromotion"]["remainingPendingCount"] == 0
    assert json.loads((target / "RIGHTS_ATTESTATION.json").read_text())["scope"] == {
        "commercialModelUse": True,
        "rawDatasetDistribution": False,
        "training": True,
    }

    samples = [
        DetectorSample.model_validate_json(line)
        for line in (target / "annotations.jsonl").read_bytes().splitlines()
    ]
    warehouse = [
        sample
        for sample in samples
        if sample.attributes.get("sourceCollection") == "FIRST_PARTY_WAREHOUSE_CAMERA"
    ]
    assert len(warehouse) == 2
    assert {len(sample.annotations) for sample in warehouse} == {0, 1}
    assert all(sample.group_id.startswith("phins-warehouse:") for sample in warehouse)
    assert all(
        sample.attributes["groupingBasis"] == "WAREHOUSE_TRANSACTION_ID" for sample in warehouse
    )
    assert (base / "source-manifest.json").read_bytes() == base_before
    assert (review / "source-manifest.json").read_bytes() == review_before

    reused = AttestedWarehouseReviewPromotionBuilder(
        base_source_directory=base,
        review_source_directory=review,
        output_directory=target,
        target_source_id="production-v2",
        decisions=decisions,
        rights_holder="duyhuynh",
        attested_by="duyhuynh",
    ).build()
    assert reused.reused is True


def _base_source(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    (labels / "images").mkdir(parents=True)
    data = _plain_jpeg(70, 1)
    (labels / "images/base.jpg").write_bytes(data)
    (labels / "annotations.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "base-label",
                "imagePath": "images/base.jpg",
                "groupId": "base-group",
                "cameraId": "base-camera",
                "capturedAt": "2026-08-01T00:00:00Z",
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 25, "y": 30, "width": 70, "height": 25},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "base.jpg").write_bytes(data)
    target = tmp_path / "sources/production-v1"
    FirstPartyPlateSourceBuilder(
        input_directory=collected,
        output_directory=target,
        label_reference_directory=labels,
        source_id="production-v1",
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    return target


def _review_source(tmp_path: Path) -> Path:
    archive = tmp_path / "warehouse.tar.gz"
    _archive(
        archive,
        [
            (
                _name("11111111-1111-1111-1111-111111111111", "a"),
                _warehouse_jpeg(90, reverse=False),
            ),
            (
                _name("22222222-2222-2222-2222-222222222222", "b"),
                _warehouse_jpeg(135, reverse=True),
            ),
        ],
    )
    target = tmp_path / "sources/warehouse-review-v1"
    WarehousePlateReviewSourceBuilder(
        archive_path=archive,
        output_directory=target,
        source_id="warehouse-review-v1",
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    return target


def _decisions(review: Path) -> dict[str, DetectorReviewDecision]:
    records = [
        json.loads(line) for line in (review / "REVIEW_QUEUE.jsonl").read_text().splitlines()
    ]
    records.sort(key=lambda item: item["reviewId"])
    reviewed_at = datetime(2026, 8, 27, 7, 0, tzinfo=UTC)
    result: dict[str, DetectorReviewDecision] = {}
    for index, record in enumerate(records):
        if index == 0:
            decision = DetectorReviewDecision(
                action=DetectorReviewAction.CORRECT,
                status=DetectorReviewStatus.CORRECTED,
                annotations=(
                    DetectorReviewAnnotation(
                        bbox=DetectorReviewBox(x=35, y=40, width=55, height=22),
                    ),
                ),
                revision=1,
                reviewed_by="operator",
                reviewer_display_name="Operator",
                reviewed_at=reviewed_at,
            )
        else:
            decision = DetectorReviewDecision(
                action=DetectorReviewAction.MARK_NEGATIVE,
                status=DetectorReviewStatus.NEGATIVE,
                annotations=(),
                revision=1,
                reviewed_by="operator",
                reviewer_display_name="Operator",
                reviewed_at=reviewed_at,
            )
        result[record["reviewId"]] = decision
    return result


def _archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    with tarfile.open(path, "w:gz") as stream:
        directory = tarfile.TarInfo("Cameras")
        directory.type = tarfile.DIRTYPE
        stream.addfile(directory)
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            stream.addfile(info, io.BytesIO(data))


def _name(group_id: str, nonce: str) -> str:
    return f"Cameras/front_image_{group_id}_1787698282859_{nonce}.jpg"


def _warehouse_jpeg(fill: int, *, reverse: bool) -> bytes:
    image = np.full((100, 140, 3), fill, dtype=np.uint8)
    start, end = ((25, 70), (105, 25)) if reverse else ((25, 25), (105, 70))
    cv2.line(image, start, end, (240, 240, 240), 5)
    cv2.rectangle(image, (18, 12), (125, 88), (255, 0, 0), 3)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return encoded.tobytes()


def _plain_jpeg(fill: int, marker: int) -> bytes:
    image = np.full((100, 140, 3), fill, dtype=np.uint8)
    image[marker, marker] = (fill + 1, fill + 2, fill + 3)
    cv2.rectangle(image, (25, 30), (95, 55), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
