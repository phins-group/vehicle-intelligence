from __future__ import annotations

import json
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
from vehicle_intelligence.training.video_review_promotion import (
    AttestedVideoReviewPromotionBuilder,
)
from vehicle_intelligence.training.video_review_source import VideoPlateReviewSourceBuilder


def test_attested_video_review_promotion_preserves_groups_and_rights_evidence(
    tmp_path: Path,
) -> None:
    base = _base_source(tmp_path)
    review = _review_source(tmp_path)
    decisions = _decisions(review)
    base_before = (base / "source-manifest.json").read_bytes()
    review_before = (review / "source-manifest.json").read_bytes()

    with pytest.raises(DetectorDatasetError, match="requires all review decisions"):
        AttestedVideoReviewPromotionBuilder(
            base_source_directory=base,
            review_source_directory=review,
            output_directory=tmp_path / "sources/incomplete-v2",
            target_source_id="incomplete-v2",
            decisions=dict(list(decisions.items())[:1]),
            rights_holder="duyhuynh",
            attested_by="duyhuynh",
        ).build()

    target = tmp_path / "sources/production-v2"
    result = AttestedVideoReviewPromotionBuilder(
        base_source_directory=base,
        review_source_directory=review,
        output_directory=target,
        target_source_id="production-v2",
        decisions=decisions,
        rights_holder="duyhuynh",
        attested_by="duyhuynh",
        clock=lambda: datetime(2026, 8, 12, 6, 0, tzinfo=UTC),
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
    assert manifest["rightsAssertion"] == "USER_CONFIRMED_FIRST_PARTY_VIDEO_COLLECTION"
    assert manifest["rightsAttestation"]["rightsHolder"] == "duyhuynh"
    assert manifest["videoReviewPromotion"]["remainingPendingCount"] == 0
    assert json.loads((target / "RIGHTS_ATTESTATION.json").read_text())["scope"] == {
        "commercialModelUse": True,
        "rawDatasetDistribution": False,
        "training": True,
    }

    samples = [
        DetectorSample.model_validate_json(line)
        for line in (target / "annotations.jsonl").read_text().splitlines()
    ]
    video_samples = [
        sample
        for sample in samples
        if sample.attributes.get("sourceCollection") == "FIRST_PARTY_USER_COLLECTED_VIDEO"
    ]
    assert len(video_samples) == 2
    assert len({sample.group_id for sample in video_samples}) == 1
    assert {len(sample.annotations) for sample in video_samples} == {0, 1}
    assert all(
        sample.attributes["groupingBasis"] == "SOURCE_VIDEO_SHA256" for sample in video_samples
    )
    assert (base / "source-manifest.json").read_bytes() == base_before
    assert (review / "source-manifest.json").read_bytes() == review_before


def _base_source(tmp_path: Path) -> Path:
    labels = tmp_path / "labels"
    (labels / "images").mkdir(parents=True)
    data = _jpg(70, 1)
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
                        "bbox": {"x": 25, "y": 30, "width": 80, "height": 28},
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
    extraction = tmp_path / "extract"
    images = extraction / "plate/images"
    images.mkdir(parents=True)
    video_sha = "a" * 64
    records: list[dict[str, object]] = []
    for index, fill in enumerate((110, 150), start=1):
        name = f"video-{index}.jpg"
        (images / name).write_bytes(_jpg(fill, index + 1))
        records.append(
            {
                "sampleId": f"video-sample-{index}",
                "imagePath": f"images/{name}",
                "groupId": "video-original-group",
                "cameraId": "video-camera",
                "capturedAt": f"2026-08-12T04:00:0{index}Z",
                "split": None,
                "attributes": {
                    "annotationSource": "MODEL_SUGGESTION",
                    "reviewStatus": "PENDING_REVIEW",
                    "licenseReviewStatus": "REVIEW_REQUIRED",
                    "acceptanceEligible": False,
                    "releaseEligible": False,
                    "sourceVideo": "traffic.mp4",
                    "sourceVideoSha256": video_sha,
                    "sourceFrameIndex": index * 30,
                    "sourceOffsetSeconds": float(index),
                    "lighting": "DAY",
                },
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 25, "y": 30, "width": 80, "height": 28},
                        "attributes": {
                            "annotationSource": "MODEL_SUGGESTION",
                            "reviewStatus": "PENDING_REVIEW",
                            "confidence": 0.8,
                            "modelName": "plate-test",
                            "modelVersion": "v1",
                            "modelHash": "b" * 64,
                        },
                    }
                ],
            }
        )
    (extraction / "plate/annotations.auto.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (extraction / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": "VIDEO_DETECTOR_SAMPLE_EXTRACTION",
                "ownerNamespace": "phins-group",
                "founderId": "duyhuynh",
                "createdAt": "2026-08-12T04:00:00Z",
                "status": "COMPLETE",
                "sourceDirectoryName": "traffic",
                "licenseReviewStatus": "REVIEW_REQUIRED",
                "acceptanceEligible": False,
                "releaseEligible": False,
                "distributionEligible": False,
                "statistics": {"plateTrainingImages": 2},
                "sources": [
                    {
                        "sourceId": "video-a",
                        "path": "traffic.mp4",
                        "sha256": video_sha,
                        "status": "PROCESSED",
                        "licenseReviewStatus": "REVIEW_REQUIRED",
                        "releaseEligible": False,
                        "distributionEligible": False,
                    }
                ],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    target = tmp_path / "sources/video-review-v1"
    VideoPlateReviewSourceBuilder(
        extraction_directory=extraction,
        output_directory=target,
        source_id="video-review-v1",
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    return target


def _decisions(review: Path) -> dict[str, DetectorReviewDecision]:
    records = [
        json.loads(line) for line in (review / "REVIEW_QUEUE.jsonl").read_text().splitlines()
    ]
    records.sort(key=lambda item: item["reviewId"])
    reviewed_at = datetime(2026, 8, 12, 5, 0, tzinfo=UTC)
    result: dict[str, DetectorReviewDecision] = {}
    for index, record in enumerate(records):
        if index == 0:
            suggestion = record["suggestions"][0]
            bbox = suggestion["bbox"]
            decision = DetectorReviewDecision(
                action=DetectorReviewAction.APPROVE,
                status=DetectorReviewStatus.APPROVED,
                annotations=(
                    DetectorReviewAnnotation(
                        bbox=DetectorReviewBox(
                            x=bbox["x"],
                            y=bbox["y"],
                            width=bbox["width"],
                            height=bbox["height"],
                        ),
                        attributes=suggestion["attributes"],
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


def _jpg(fill: int, marker: int) -> bytes:
    image = np.full((100, 150, 3), fill, dtype=np.uint8)
    image[marker, marker] = (fill + 1, fill + 2, fill + 3)
    cv2.rectangle(image, (25, 30), (105, 58), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
