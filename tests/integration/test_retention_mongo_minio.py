import os
import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.retention import RetentionService
from vehicle_intelligence.config import MinioConfig, MongoConfig, RetentionConfig
from vehicle_intelligence.domain import MediaKind
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.retention_mongo import (
    MongoRetentionRepository,
)
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage


@pytest.mark.skipif(
    not os.getenv("TEST_MONGODB_URI") or not os.getenv("TEST_MINIO_ENDPOINT"),
    reason="TEST_MONGODB_URI and TEST_MINIO_ENDPOINT are not configured",
)
@pytest.mark.asyncio
async def test_retention_coordinates_minio_dataset_pins_and_event_deletion(
    sample_event,
) -> None:
    suffix = uuid.uuid4().hex
    now = datetime.now(UTC)
    mongodb = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    minio = MinioConfig(
        endpoint=os.environ["TEST_MINIO_ENDPOINT"],
        access_key=os.getenv("TEST_MINIO_ACCESS_KEY", "minioadmin"),
        secret_key=os.getenv("TEST_MINIO_SECRET_KEY", "minioadmin"),
        bucket="vehicle-media-test",
    )
    keys = {
        "snapshot_key": f"vehicles/retention/{suffix}/snapshot.jpg",
        "vehicle_crop_key": f"vehicles/retention/{suffix}/vehicle.jpg",
        "plate_crop_key": f"vehicles/retention/{suffix}/plate.jpg",
        "clip_key": f"vehicles/retention/{suffix}/event.mp4",
    }
    event = replace(
        sample_event,
        id=f"evt-retention-{suffix}",
        track_id=f"gate-01:retention:{suffix}",
        occurred_at=now - timedelta(days=3),
        created_at=now - timedelta(days=3),
        media=replace(sample_event.media, **keys),
    )
    event_repository = MongoVehicleEventRepository(mongodb)
    retention_repository = MongoRetentionRepository(mongodb)
    storage = MinioMediaStorage(minio)
    service = RetentionService(
        RetentionConfig(
            enabled=True,
            batch_size=20,
            vehicle_events_days=2,
            snapshots_days=1,
            vehicle_crops_days=1,
            plate_crops_days=1,
            event_clips_days=1,
            debug_images_days=1,
        ),
        retention_repository,
        storage,
        lifecycle=storage,
        clock=lambda: now,
    )
    try:
        await event_repository.ensure_indexes()
        assert await event_repository.save(event)
        for key in keys.values():
            await storage.put(key, b"retention-evidence", "application/octet-stream")
        await retention_repository._samples.insert_one(
            {
                "_id": f"dss-retention-{suffix}",
                "sourceEventId": event.id,
                "imageKey": keys["snapshot_key"],
                "status": "READY",
                "createdAt": now,
            }
        )

        first = await service.run_once()
        retained = await event_repository.get(event.id)

        assert first.lifecycle is not None
        assert first.lifecycle.managed_rules == 2
        assert sum(first.deleted_by_kind.values()) == 3
        assert retained is not None
        assert retained.media.snapshot_key == keys["snapshot_key"]
        assert retained.media.vehicle_crop_key is None
        assert retained.media.clip_key is None
        assert retained.media.plate_crop_key is None
        assert await storage.exists(keys["snapshot_key"])

        await retention_repository._samples.update_one(
            {"sourceEventId": event.id},
            {"$set": {"status": "EXPORTING"}},
        )
        exporting = await service.run_once()
        await retention_repository._samples.update_one(
            {"sourceEventId": event.id},
            {"$set": {"status": "EXPORT_FAILED"}},
        )
        failed = await service.run_once()

        assert exporting.deleted_by_kind == {}
        assert failed.deleted_by_kind == {}
        assert await storage.exists(keys["snapshot_key"])

        await retention_repository._samples.update_one(
            {"sourceEventId": event.id},
            {"$set": {"status": "EXPORTED"}},
        )
        second = await service.run_once()

        assert second.deleted_by_kind == {"snapshot": 1}
        assert second.events_deleted == 1
        assert await event_repository.get(event.id) is None
        remaining_objects = [await storage.exists(key) for key in keys.values()]
        assert remaining_objects == [False, False, False, False]
        index_cursor = await retention_repository._samples.list_indexes()
        indexes = {item["name"] async for item in index_cursor}
        assert "ix_dataset_image_status" in indexes
    finally:
        await retention_repository._samples.delete_many({"sourceEventId": event.id})
        await event_repository._collection.delete_one({"_id": event.id})
        for key in keys.values():
            await storage.remove(key)
        await service.close()
        await event_repository.close()


@pytest.mark.skipif(
    not os.getenv("TEST_MONGODB_URI"),
    reason="TEST_MONGODB_URI is not configured",
)
@pytest.mark.asyncio
async def test_retention_lease_reclaims_stale_work_and_restores_failed_media() -> None:
    suffix = uuid.uuid4().hex
    now = datetime.now(UTC)
    mongodb = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    repository = MongoRetentionRepository(mongodb)
    event_id = f"evt-retention-lease-{suffix}"
    key = f"vehicles/retention/{suffix}/snapshot.jpg"
    try:
        await repository.ensure_indexes()
        await repository._events.insert_one(
            {
                "_id": event_id,
                "occurredAt": now - timedelta(days=2),
                "media": {"snapshotKey": key},
            }
        )

        first = await repository.claim_media(
            MediaKind.SNAPSHOT,
            now - timedelta(days=1),
            now - timedelta(minutes=1),
            "lease-one",
            1,
        )
        concurrent = await repository.claim_media(
            MediaKind.SNAPSHOT,
            now - timedelta(days=1),
            now - timedelta(minutes=1),
            "lease-concurrent",
            1,
        )
        assert len(first) == 1
        assert concurrent == []

        await repository._events.update_one(
            {"_id": event_id},
            {"$set": {"retention.media.snapshot.updatedAt": now - timedelta(minutes=2)}},
        )
        reclaimed = await repository.claim_media(
            MediaKind.SNAPSHOT,
            now - timedelta(days=1),
            now - timedelta(minutes=1),
            "lease-two",
            1,
        )
        assert len(reclaimed) == 1
        assert reclaimed[0].lease_id == "lease-two"

        await repository.mark_media_failed(
            reclaimed[0],
            "StorageUnavailable",
            now,
        )
        failed = await repository._events.find_one({"_id": event_id})
        assert failed["media"]["snapshotKey"] == key
        assert failed["retention"]["media"]["snapshot"]["state"] == "FAILED"

        retried = await repository.claim_media(
            MediaKind.SNAPSHOT,
            now - timedelta(days=1),
            now - timedelta(minutes=1),
            "lease-three",
            1,
        )
        await repository.mark_media_deleted(retried[0], now)
        deleted = await repository._events.find_one({"_id": event_id})
        assert "snapshotKey" not in deleted["media"]
        assert deleted["retention"]["media"]["snapshot"]["state"] == "DELETED"
    finally:
        await repository._events.delete_one({"_id": event_id})
        await repository.close()
