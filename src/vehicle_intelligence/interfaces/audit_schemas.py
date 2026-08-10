"""Public audit-log schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.domain import AuditLog


class APIModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class AuditActorPublic(APIModel):
    id: str
    display_name: str = Field(alias="displayName")
    role: str
    authentication_method: str = Field(alias="authenticationMethod")


class AuditResourcePublic(APIModel):
    type: str
    id: str


class AuditLogPublic(APIModel):
    id: str
    schema_version: int = Field(alias="schemaVersion")
    actor: AuditActorPublic
    action: str
    resource: AuditResourcePublic
    request_id: str = Field(alias="requestId")
    before: dict[str, object] | None
    after: dict[str, object] | None
    metadata: dict[str, object]
    occurred_at: datetime = Field(alias="occurredAt")

    @classmethod
    def from_domain(cls, entry: AuditLog) -> AuditLogPublic:
        return cls(
            id=entry.id,
            schemaVersion=entry.schema_version,
            actor=AuditActorPublic(
                id=entry.actor.id,
                displayName=entry.actor.display_name,
                role=entry.actor.role.value,
                authenticationMethod=entry.actor.authentication_method.value,
            ),
            action=entry.action.value,
            resource=AuditResourcePublic(
                type=entry.resource_type.value,
                id=entry.resource_id,
            ),
            requestId=entry.request_id,
            before=entry.before,
            after=entry.after,
            metadata=entry.metadata,
            occurredAt=entry.occurred_at,
        )


class AuditLogListPublic(APIModel):
    items: list[AuditLogPublic]
    next_cursor: str | None = Field(alias="nextCursor")

