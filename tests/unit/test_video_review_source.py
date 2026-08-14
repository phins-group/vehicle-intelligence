from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.application.dataset_review import (
    DetectorDatasetReviewService,
    DetectorReviewCommand,
    DetectorReviewQuery,
)
from vehicle_intelligence.config import DatasetReviewConfig
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    Principal,
    UserRole,
)
from vehicle_intelligence.domain.dataset_review import (
    DetectorReviewAction,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import DatasetReviewValidationError
from vehicle_intelligence.infrastructure.training.dataset_review_files import (
    FileDetectorReviewRepository,
)
from vehicle_intelligence.training.video_review_source import (
    VIDEO_REVIEW_REASON,
    VIDEO_REVIEW_SOURCE_TYPE,
    VideoPlateReviewSourceBuilder,
    verify_video_plate_review_source,
)


@pytest.mark.asyncio
async def test_video_extraction_becomes_review_only_source_with_deduplicated_images(
    tmp_path: Path,
) -> None:
    extraction = _extraction(tmp_path)
    source_id = "phins-video-review-test-v1"
    sources = tmp_path / "sources"
    result = VideoPlateReviewSourceBuilder(
        extraction_directory=extraction,
        output_directory=sources / source_id,
        source_id=source_id,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
        clock=lambda: datetime(2026, 8, 12, 5, 0, tzinfo=UTC),
    ).build()

    assert result.source_record_count == 2
    assert result.review_queue_count == 1
    assert result.suggestion_count == 1
    assert result.exact_duplicate_images_merged == 1
    manifest, digest = verify_video_plate_review_source(result.directory)
    assert digest == result.manifest_sha256
    assert manifest["type"] == VIDEO_REVIEW_SOURCE_TYPE
    assert manifest["promotionEligible"] is False
    assert manifest["releaseEligible"] is False
    assert manifest["statistics"]["exactDuplicateImagesMerged"] == 1
    assert len(list((result.directory / "review/images").rglob("*.jpg"))) == 1

    config = DatasetReviewConfig(
        sources_directory=sources,
        workspace_directory=tmp_path / "reviews",
        promoted_sources_directory=sources,
    )
    repository = FileDetectorReviewRepository(config)
    service = DetectorDatasetReviewService(repository)
    await service.initialize()
    summary = (await service.list_sources())[0]
    assert summary.source_type == VIDEO_REVIEW_SOURCE_TYPE
    assert summary.rights_status == "REVIEW_REQUIRED"
    assert summary.promotion_eligible is False
    assert summary.pending_count == 1

    page = await service.list_items(DetectorReviewQuery(source_id=source_id))
    assert len(page.items) == 1
    item = page.items[0]
    assert item.reason == VIDEO_REVIEW_REASON
    assert len(item.suggestions) == 1
    assert item.suggestions[0].attributes["confidence"] == 0.82
    reviewed = await service.review(
        source_id,
        item.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    assert reviewed.status is DetectorReviewStatus.APPROVED
    with pytest.raises(DatasetReviewValidationError, match="not eligible for promotion"):
        await repository.create_promotion_job(source_id, "video-review-production-v2", "admin")
    await service.close()


def _extraction(tmp_path: Path) -> Path:
    root = tmp_path / "extract"
    images = root / "plate/images"
    images.mkdir(parents=True)
    image = np.full((100, 160, 3), 90, dtype=np.uint8)
    cv2.rectangle(image, (30, 40), (120, 70), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    data = encoded.tobytes()
    (images / "sample-a.jpg").write_bytes(data)
    (images / "sample-b.jpg").write_bytes(data)
    video_sha = "a" * 64
    records = [
        _sample_record("sample-a", "images/sample-a.jpg", video_sha, 0.71),
        _sample_record("sample-b", "images/sample-b.jpg", video_sha, 0.82),
    ]
    (root / "plate/annotations.auto.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": "VIDEO_DETECTOR_SAMPLE_EXTRACTION",
                "ownerNamespace": "phins-group",
                "founderId": "duyhuynh",
                "createdAt": "2026-08-12T04:00:00Z",
                "status": "COMPLETE",
                "sourceDirectoryName": "vn_traffic",
                "licenseReviewStatus": "REVIEW_REQUIRED",
                "acceptanceEligible": False,
                "releaseEligible": False,
                "distributionEligible": False,
                "statistics": {"plateTrainingImages": 2},
                "sources": [
                    {
                        "sourceId": "video-source-a",
                        "path": "traffic.mp4",
                        "sha256": video_sha,
                        "status": "PROCESSED",
                        "licenseReviewStatus": "REVIEW_REQUIRED",
                        "releaseEligible": False,
                        "distributionEligible": False,
                    }
                ],
                "models": {
                    "plate": {
                        "provider": "test",
                        "name": "plate-test",
                        "version": "v1",
                        "sha256": "b" * 64,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def _sample_record(
    sample_id: str,
    image_path: str,
    video_sha: str,
    confidence: float,
) -> dict[str, object]:
    return {
        "sampleId": sample_id,
        "imagePath": image_path,
        "groupId": "video-group-a",
        "cameraId": "video-camera-a",
        "capturedAt": "2026-08-12T04:00:00Z",
        "split": None,
        "attributes": {
            "annotationSource": "MODEL_SUGGESTION",
            "reviewStatus": "PENDING_REVIEW",
            "licenseReviewStatus": "REVIEW_REQUIRED",
            "acceptanceEligible": False,
            "releaseEligible": False,
            "sourceVideo": "traffic.mp4",
            "sourceVideoSha256": video_sha,
            "sourceFrameIndex": 30,
            "sourceOffsetSeconds": 1.0,
            "lighting": "DAY",
        },
        "annotations": [
            {
                "className": "license_plate",
                "bbox": {"x": 30, "y": 40, "width": 90, "height": 30},
                "attributes": {
                    "annotationSource": "MODEL_SUGGESTION",
                    "reviewStatus": "PENDING_REVIEW",
                    "confidence": confidence,
                    "modelName": "plate-test",
                    "modelVersion": "v1",
                    "modelHash": "b" * 64,
                    "cropPath": f"crops/{sample_id}.jpg",
                },
            }
        ],
    }


def _operator() -> Principal:
    return Principal(
        id="operator-01",
        display_name="Dataset Operator",
        role=UserRole.OPERATOR,
        authentication_method=AuthenticationMethod.DEVELOPMENT,
    )
