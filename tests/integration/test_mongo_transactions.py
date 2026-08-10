import base64
import os
import uuid
from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.config import MongoConfig, SecurityConfig
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    AuthenticationMethod,
    Camera,
    CameraDirection,
    Principal,
    SecretUri,
    UserRole,
)
from vehicle_intelligence.exceptions import AuditWriteError
from vehicle_intelligence.infrastructure.persistence.audit_mongo import MongoAuditLogRepository
from vehicle_intelligence.infrastructure.persistence.camera_mongo import MongoCameraRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_resource_and_required_audit_commit_or_rollback_atomically() -> None:
    suffix = uuid.uuid4().hex
    camera_id = f"tx-camera-{suffix}"
    audit_id = f"tx-audit-{suffix}"
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
        transactions_enabled=True,
    )
    runtime = MongoRuntime(config)
    await runtime.initialize()
    cipher = AesGcmCredentialCipher.from_config(
        SecurityConfig(
            camera_credential_key=base64.urlsafe_b64encode(bytes(range(32))).decode(),
            camera_credential_key_id="test",
        )
    )
    cameras = MongoCameraRepository(runtime, cipher)
    audit_repository = MongoAuditLogRepository(runtime)
    timestamp = datetime(2026, 8, 10, tzinfo=UTC)
    principal = Principal(
        id="tx-admin",
        display_name="Transaction Admin",
        role=UserRole.ADMIN,
        authentication_method=AuthenticationMethod.API_KEY,
    )
    camera = Camera(
        id=camera_id,
        name="Transaction Gate",
        rtsp_url=SecretUri("rtsp://camera/live"),
        fps_limit=6,
        direction=CameraDirection.BOTH,
        enabled=True,
        vehicle_confidence=0.4,
        plate_confidence=0.45,
        created_at=timestamp,
        updated_at=timestamp,
    )
    command = AuditRecord(
        principal=principal,
        action=AuditAction.CAMERA_CREATED,
        resource_type=AuditResourceType.CAMERA,
        resource_id=camera_id,
        request_id=f"request-{suffix}",
    )
    duplicate_audit = AuditService(
        audit_repository,
        clock=lambda: timestamp,
        id_factory=lambda: audit_id,
    )
    try:
        await cameras.ensure_indexes()
        await duplicate_audit.initialize()
        await duplicate_audit.record(command)

        with pytest.raises(AuditWriteError):
            async with runtime.transaction():
                assert await cameras.create(camera)
                await duplicate_audit.record(command)
        assert await cameras.get(camera_id) is None

        successful_audit = AuditService(
            audit_repository,
            clock=lambda: timestamp,
            id_factory=lambda: f"{audit_id}-committed",
        )
        async with runtime.transaction():
            assert await cameras.create(camera)
            committed = await successful_audit.record(command)
        assert await cameras.get(camera_id) == camera
        assert await audit_repository.get(committed.id) is not None
    finally:
        await cameras._collection.delete_one({"_id": camera_id})
        await audit_repository._collection.delete_many({"_id": {"$regex": f"^{audit_id}"}})
        await duplicate_audit.close()
        await cameras.close()
        await runtime.close()
