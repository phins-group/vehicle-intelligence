import os
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.ports import AuditQuery
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    AuthenticationMethod,
    Principal,
    UserRole,
)
from vehicle_intelligence.infrastructure.persistence.audit_mongo import MongoAuditLogRepository


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is not configured")
async def test_mongo_audit_log_is_append_only_indexed_and_redacted() -> None:
    suffix = uuid.uuid4().hex
    entry_ids = iter((f"aud-{suffix}-1", f"aud-{suffix}-2"))
    config = MongoConfig(
        enabled=True,
        uri=os.environ["TEST_MONGODB_URI"],
        database="vehicle_intelligence_test",
    )
    repository = MongoAuditLogRepository(config)
    timestamp = datetime(2026, 8, 9, 12, tzinfo=UTC)
    times = iter((timestamp, timestamp + timedelta(seconds=1)))
    service = AuditService(
        repository,
        clock=lambda: next(times),
        id_factory=lambda: next(entry_ids),
    )
    principal = Principal(
        id="admin-mongo",
        display_name="Mongo Admin",
        role=UserRole.ADMIN,
        authentication_method=AuthenticationMethod.API_KEY,
    )
    try:
        await service.initialize()
        first = await service.record(
            AuditRecord(
                principal=principal,
                action=AuditAction.CAMERA_CREATED,
                resource_type=AuditResourceType.CAMERA,
                resource_id=f"gate-{suffix}",
                request_id=f"req-{suffix}-1",
                after={"name": "Gate", "rtspUrl": "rtsp://admin:secret@example/live"},
            )
        )
        second = await service.record(
            AuditRecord(
                principal=principal,
                action=AuditAction.CAMERA_DISABLED,
                resource_type=AuditResourceType.CAMERA,
                resource_id=f"gate-{suffix}",
                request_id=f"req-{suffix}-2",
                before={"enabled": True},
                after={"enabled": False},
            )
        )

        page = await service.list(
            AuditQuery(
                actor_id=principal.id,
                resource_type=AuditResourceType.CAMERA,
                resource_id=f"gate-{suffix}",
                limit=10,
            )
        )
        raw = await repository._collection.find_one({"_id": first.id})
        indexes = {item["name"] async for item in await repository._collection.list_indexes()}

        assert page.items == (second, first)
        assert raw["after"]["rtspUrl"] == "[REDACTED]"
        assert "secret" not in str(raw)
        assert {
            "ix_audit_cursor",
            "ix_audit_actor_time",
            "ix_audit_resource_time",
            "ix_audit_action_time",
        } <= indexes
    finally:
        await repository._collection.delete_many({"resource.id": f"gate-{suffix}"})
        await service.close()

