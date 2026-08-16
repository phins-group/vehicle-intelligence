import asyncio
import base64
import os
import uuid
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.ports import CameraCreateOutcome
from vehicle_intelligence.config import MongoConfig, SecurityConfig
from vehicle_intelligence.domain import (
    Camera,
    CameraDirection,
    CameraHealth,
    CameraStatus,
    SecretUri,
)
from vehicle_intelligence.infrastructure.persistence.camera_mongo import (
    MongoCameraHealthRepository,
    MongoCameraRepository,
)
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_camera_credentials_are_encrypted_and_health_is_latest_state() -> None:
    suffix = uuid.uuid4().hex
    camera_id = f"gate-mongo-{suffix}"
    capacity_camera_ids = (f"capacity-a-{suffix}", f"capacity-b-{suffix}")
    secret = f"mongo-secret-{suffix}"
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    key = base64.urlsafe_b64encode(bytes(range(32))).decode()
    cipher = AesGcmCredentialCipher.from_config(
        SecurityConfig(camera_credential_key=key, camera_credential_key_id="test-key")
    )
    repository = MongoCameraRepository(config, cipher)
    health_repository = MongoCameraHealthRepository(config)
    timestamp = datetime(2026, 8, 9, tzinfo=UTC)
    camera = Camera(
        id=camera_id,
        name="Mongo Gate",
        rtsp_url=SecretUri(f"rtsp://admin:{secret}@camera.example/live"),
        fps_limit=6,
        direction=CameraDirection.BOTH,
        enabled=True,
        vehicle_confidence=0.4,
        plate_confidence=0.45,
        created_at=timestamp,
        updated_at=timestamp,
    )
    try:
        await repository.ensure_indexes()
        await health_repository.ensure_indexes()
        initial_camera_count = await repository.count()
        assert await repository.create(camera)
        assert await repository.count() == initial_camera_count + 1
        raw = await repository._collection.find_one({"_id": camera_id})
        encrypted = raw["stream"]["rtspUrlEncrypted"]
        assert secret not in encrypted
        assert "rtspUrl" not in raw["stream"]
        assert encrypted.startswith("v1.test-key.")
        assert await repository.get(camera_id) == camera

        capacity_limit = await repository.count() + 1
        capacity_outcomes = await asyncio.gather(
            *(
                repository.create_with_capacity(
                    replace(camera, id=candidate_id, name=f"Capacity {candidate_id}"),
                    capacity_limit,
                )
                for candidate_id in capacity_camera_ids
            )
        )
        assert capacity_outcomes.count(CameraCreateOutcome.CREATED) == 1
        assert capacity_outcomes.count(CameraCreateOutcome.CAPACITY_REACHED) == 1
        assert await repository.count() == capacity_limit

        updated = replace(camera, name="Updated Mongo Gate", revision=2)
        assert await repository.replace(updated, 1)
        rotated = await repository._collection.find_one({"_id": camera_id})
        assert rotated["stream"]["rtspUrlEncrypted"] != encrypted

        online = CameraHealth(
            camera_id=camera_id,
            status=CameraStatus.ONLINE,
            source_fps=25,
            decode_fps=24,
            queue_size=1,
            dropped_frames=3,
            reconnect_count=1,
            connection_failures=1,
            stream_epoch=1,
            last_frame_at=timestamp,
            updated_at=timestamp,
            decoded_frames=120,
            sampled_frames=30,
            vehicle_detections=12,
            plate_detections=8,
            ocr_requests=6,
            ocr_success=5,
            events_created=3,
            track_count=2,
            inference_fps=6.1,
            vehicle_inference_latency_ms=14.2,
            plate_inference_latency_ms=7.4,
            ocr_latency_ms=21.5,
        )
        await health_repository.save(online)
        await health_repository.save(replace(online, status=CameraStatus.OFFLINE))
        assert await health_repository._collection.count_documents({"_id": camera_id}) == 1
        persisted_health = await health_repository.get(camera_id)
        assert persisted_health is not None
        assert persisted_health.status is CameraStatus.OFFLINE
        assert persisted_health.decoded_frames == 120
        assert persisted_health.ocr_success == 5
        assert persisted_health.inference_fps == 6.1
        assert persisted_health.ocr_latency_ms == 21.5
    finally:
        for persisted_camera_id in (camera_id, *capacity_camera_ids):
            await repository.delete(persisted_camera_id)
        await health_repository._collection.delete_one({"_id": camera_id})
        await repository.close()
        await health_repository.close()
