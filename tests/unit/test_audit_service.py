from datetime import UTC, datetime

import pytest

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.ports import AuditQuery
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    AuthenticationMethod,
    Principal,
    UserRole,
)
from vehicle_intelligence.exceptions import AuditWriteError, PersistenceError
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)


def principal() -> Principal:
    return Principal(
        id="admin-01",
        display_name="Platform Admin",
        role=UserRole.ADMIN,
        authentication_method=AuthenticationMethod.API_KEY,
    )


async def test_audit_service_redacts_credentials_and_keeps_safe_context() -> None:
    timestamp = datetime(2026, 8, 9, 12, tzinfo=UTC)
    repository = InMemoryAuditLogRepository()
    service = AuditService(
        repository,
        clock=lambda: timestamp,
        id_factory=lambda: "aud-test",
    )

    entry = await service.record(
        AuditRecord(
            principal=principal(),
            action=AuditAction.CAMERA_CREATED,
            resource_type=AuditResourceType.CAMERA,
            resource_id="gate-01",
            request_id="req-test",
            after={
                "name": "Main Gate",
                "rtspUrl": "rtsp://admin:camera-secret@example/live",
                "nested": {"password": "do-not-store", "zone": "ZONE_A"},
                "authorization": "Bearer raw-token",
                "webhook": "https://hooks.example/vehicle?token=query-secret",
                "headers": {"X-API-Key": "header-secret"},
            },
        )
    )

    assert entry.after["rtspUrl"] == "[REDACTED]"
    assert entry.after["nested"]["password"] == "[REDACTED]"
    assert entry.after["nested"]["zone"] == "ZONE_A"
    assert entry.after["authorization"] == "[REDACTED]"
    assert entry.after["webhook"] == "https://hooks.example/vehicle"
    assert entry.after["headers"]["X-API-Key"] == "[REDACTED]"
    page = await service.list(AuditQuery(resource_id="gate-01"))
    assert page.items == (entry,)


class FailingAuditRepository(InMemoryAuditLogRepository):
    async def append(self, entry) -> None:
        raise PersistenceError(f"cannot append {entry.id}")


async def test_audit_write_failure_is_explicit() -> None:
    service = AuditService(FailingAuditRepository(), id_factory=lambda: "aud-failed")
    with pytest.raises(AuditWriteError, match="could not be persisted"):
        await service.record(
            AuditRecord(
                principal=principal(),
                action=AuditAction.RULE_CREATED,
                resource_type=AuditResourceType.RULE,
                resource_id="rule-01",
                request_id="req-failed",
            )
        )
