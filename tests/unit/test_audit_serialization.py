from datetime import UTC, datetime

from vehicle_intelligence.domain import (
    AuditAction,
    AuditActor,
    AuditLog,
    AuditResourceType,
    AuthenticationMethod,
    UserRole,
)
from vehicle_intelligence.infrastructure.audit_serialization import (
    audit_log_from_json,
    audit_log_to_json,
)


def test_audit_outbox_json_round_trip() -> None:
    entry = AuditLog(
        id="aud-round-trip",
        actor=AuditActor(
            id="operator-01",
            display_name="Operator",
            role=UserRole.OPERATOR,
            authentication_method=AuthenticationMethod.API_KEY,
        ),
        action=AuditAction.DETECTOR_SAMPLE_REVIEWED,
        resource_type=AuditResourceType.DETECTOR_DATASET_SAMPLE,
        resource_id="source:review-0123456789abcdef01234567",
        occurred_at=datetime(2026, 8, 15, 12, tzinfo=UTC),
        request_id="req-round-trip",
        before={"revision": 0, "tags": ["review"]},
        after={"revision": 1, "status": "APPROVED"},
        metadata={"reviewRevision": 1},
    )

    assert audit_log_from_json(audit_log_to_json(entry)) == entry
