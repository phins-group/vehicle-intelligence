from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.enums import (
    AuditAction,
    AuditResourceType,
    AuthenticationMethod,
    UserRole,
)


@dataclass(frozen=True, slots=True)
class AuditActor:
    id: str
    display_name: str
    role: UserRole
    authentication_method: AuthenticationMethod

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.display_name.strip():
            raise ValueError("audit actor identity is required")


@dataclass(frozen=True, slots=True)
class AuditLog:
    id: str
    actor: AuditActor
    action: AuditAction
    resource_type: AuditResourceType
    resource_id: str
    occurred_at: datetime
    request_id: str
    before: dict[str, object] | None = None
    after: dict[str, object] | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        identifiers = (self.id, self.resource_id, self.request_id)
        if any(not value.strip() for value in identifiers):
            raise ValueError("audit identifiers are required")
        if self.occurred_at.tzinfo is None:
            raise ValueError("audit occurred_at must be timezone-aware")
        if self.schema_version < 1:
            raise ValueError("audit schema version must be positive")
