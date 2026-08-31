from __future__ import annotations

import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

from vehicle_intelligence.application.dataset_review import (
    DetectorDatasetReviewService,
    DetectorReviewQuery,
)
from vehicle_intelligence.config import DatasetReviewConfig
from vehicle_intelligence.domain import BoundingBox, ModelMetadata, PlateDetection
from vehicle_intelligence.infrastructure.training.dataset_review_files import (
    FileDetectorReviewRepository,
)
from vehicle_intelligence.training.review_suggestions import (
    DetectorReviewSuggestionGenerator,
    ReviewSuggestionModel,
    ReviewSuggestionOptions,
)
from vehicle_intelligence.training.warehouse_plate_review import (
    WAREHOUSE_PLATE_REVIEW_REASON,
    WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE,
    WarehousePlateReviewSourceBuilder,
    verify_warehouse_plate_review_source,
)


class _PlateDetector:
    def detect(self, _image: np.ndarray) -> list[PlateDetection]:
        return [
            PlateDetection(
                bbox=BoundingBox(35, 42, 76, 60),
                confidence=0.91,
                model=ModelMetadata("plate-test", "v1", "b" * 64),
            )
        ]


@pytest.mark.asyncio
async def test_warehouse_images_become_clean_deduplicated_plate_review_queue(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "warehouse.tar.gz"
    first = _warehouse_jpeg(80, reverse=False)
    second = _warehouse_jpeg(125, reverse=True)
    _archive(
        archive,
        [
            (_name("11111111-1111-1111-1111-111111111111", "a"), first),
            (_name("11111111-1111-1111-1111-111111111111", "b"), first),
            (_name("22222222-2222-2222-2222-222222222222", "a"), first + b"meta"),
            (_name("33333333-3333-3333-3333-333333333333", "a"), second),
            (_name("44444444-4444-4444-4444-444444444444", "a"), _plain_jpeg()),
        ],
    )
    source_id = "warehouse-plate-review-v1"
    sources = tmp_path / "sources"
    target = sources / source_id
    result = WarehousePlateReviewSourceBuilder(
        archive_path=archive,
        output_directory=target,
        source_id=source_id,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
        clock=lambda: datetime(2026, 8, 27, 4, 0, tzinfo=UTC),
    ).build()

    manifest, digest = verify_warehouse_plate_review_source(target)
    assert result.manifest_sha256 == digest
    assert result.archive_image_count == 5
    assert result.unique_raw_image_count == 4
    assert result.review_queue_count == 2
    assert result.exact_duplicate_files_excluded == 1
    assert result.near_duplicate_images_excluded == 1
    assert result.rejected_unique_images == 1
    assert manifest["type"] == WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE
    assert manifest["promotionEligible"] is False
    assert manifest["reviewQueueCount"] == 2

    duplicate_reasons = {
        json.loads(line)["reason"]
        for line in (target / "DUPLICATES.jsonl").read_text().splitlines()
    }
    assert duplicate_reasons == {
        "EXACT_SHA256_DUPLICATE",
        "PERCEPTUAL_NEAR_DUPLICATE",
    }
    queue = [json.loads(line) for line in (target / "REVIEW_QUEUE.jsonl").read_text().splitlines()]
    assert {record["reason"] for record in queue} == {WAREHOUSE_PLATE_REVIEW_REASON}
    assert all(record["suggestions"] == [] for record in queue)
    for record in queue:
        image = cv2.imread(str(target / record["imagePath"]))
        assert image is not None
        blue, green, red = cv2.split(image)
        burned_blue = (
            (blue >= 155)
            & (green <= 75)
            & (red <= 75)
            & (blue.astype(np.int16) >= green.astype(np.int16) + 90)
            & (blue.astype(np.int16) >= red.astype(np.int16) + 90)
        )
        assert int(np.count_nonzero(burned_blue)) == 0

    workspace = tmp_path / "reviews"
    suggestions = DetectorReviewSuggestionGenerator(
        _PlateDetector(),
        ReviewSuggestionModel(
            provider="test",
            name="plate-test",
            version="v1",
            sha256="b" * 64,
            confidence=0.7,
            iou=0.6,
            image_size=1280,
        ),
        ReviewSuggestionOptions(
            source_directory=target,
            workspace_directory=workspace,
        ),
    ).generate()
    assert suggestions.suggested_items == 2
    assert suggestions.suggestion_boxes == 2

    repository = FileDetectorReviewRepository(
        DatasetReviewConfig(
            sources_directory=sources,
            workspace_directory=workspace,
            promoted_sources_directory=sources,
        )
    )
    service = DetectorDatasetReviewService(repository)
    await service.initialize()
    summary = (await service.list_sources())[0]
    assert summary.source_type == WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE
    assert summary.pending_count == 2
    assert summary.promotion_eligible is False
    page = await service.list_items(DetectorReviewQuery(source_id=source_id))
    assert len(page.items) == 2
    assert all(len(item.suggestions) == 1 for item in page.items)
    await service.close()

    reused = WarehousePlateReviewSourceBuilder(
        archive_path=archive,
        output_directory=target,
        source_id=source_id,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    assert reused.reused is True


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


def _plain_jpeg() -> bytes:
    image = np.full((100, 140, 3), 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
