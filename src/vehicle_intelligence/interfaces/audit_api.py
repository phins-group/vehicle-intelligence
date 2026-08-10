"""Read-only, ADMIN-protected audit API."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.ports import AuditQuery
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.exceptions import (
    AuditNotFoundError,
    InvalidCursorError,
    PersistenceError,
)
from vehicle_intelligence.interfaces.audit_schemas import AuditLogListPublic, AuditLogPublic
from vehicle_intelligence.interfaces.security import APISecurity


def build_audit_router(audits: AuditService, security: APISecurity) -> APIRouter:
    router = APIRouter(prefix="/api/audit-logs", tags=["audit"])
    audit_access = security.require(Permission.READ_AUDIT_LOGS)

    @router.get("", response_model=AuditLogListPublic)
    async def list_audit_logs(
        _principal: Principal = Depends(audit_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        actor_id: Annotated[str | None, Query(alias="actorId")] = None,
        action: AuditAction | None = None,
        resource_type: Annotated[
            AuditResourceType | None, Query(alias="resourceType")
        ] = None,
        resource_id: Annotated[str | None, Query(alias="resourceId")] = None,
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
    ) -> AuditLogListPublic:
        _validate_time(from_time, "from")
        _validate_time(to_time, "to")
        try:
            page = await audits.list(
                AuditQuery(
                    limit=limit,
                    cursor=cursor,
                    actor_id=actor_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
        except InvalidCursorError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except PersistenceError as exc:
            raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc
        return AuditLogListPublic(
            items=[AuditLogPublic.from_domain(item) for item in page.items],
            nextCursor=page.next_cursor,
        )

    @router.get("/{entry_id}", response_model=AuditLogPublic)
    async def get_audit_log(
        entry_id: str,
        _principal: Principal = Depends(audit_access),
    ) -> AuditLogPublic:
        try:
            return AuditLogPublic.from_domain(await audits.get(entry_id))
        except AuditNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PersistenceError as exc:
            raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc

    return router


def _validate_time(value: datetime | None, field: str) -> None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(status_code=422, detail=f"{field} timestamp must include a timezone")
