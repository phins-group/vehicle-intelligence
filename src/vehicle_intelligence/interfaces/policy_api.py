"""FastAPI routes for watchlists, declarative rules, and alerts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.policies import PolicyServices
from vehicle_intelligence.application.ports import AlertQuery
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import (
    AlertStatus,
    AuditAction,
    AuditResourceType,
    Principal,
    WatchlistType,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    InvalidCursorError,
    PersistenceError,
    PolicyConflictError,
    PolicyNotFoundError,
    RuleValidationError,
)
from vehicle_intelligence.interfaces.policy_schemas import (
    AlertListPublic,
    AlertPublic,
    AlertTransitionRequest,
    RuleCreateRequest,
    RuleListPublic,
    RulePublic,
    RuleUpdateRequest,
    WatchlistCreateRequest,
    WatchlistListPublic,
    WatchlistPublic,
    WatchlistUpdateRequest,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity


def build_policy_router(
    services: PolicyServices,
    security: APISecurity,
    audits: AuditService,
    mutation_transaction: Callable[[], AsyncIterator[None]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api")
    read_access = security.require(Permission.READ_PLATFORM)
    policy_access = security.require(Permission.MANAGE_POLICIES)
    alert_access = security.require(Permission.MANAGE_ALERTS)
    mutation_dependencies = (
        [Depends(mutation_transaction)] if mutation_transaction is not None else []
    )

    @router.post(
        "/watchlists",
        response_model=WatchlistPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=mutation_dependencies,
    )
    async def create_watchlist(
        http_request: Request,
        request: WatchlistCreateRequest,
        principal: Principal = Depends(policy_access),
    ) -> WatchlistPublic:
        try:
            created = WatchlistPublic.from_domain(
                await services.watchlists.create(request.to_command())
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.WATCHLIST_CREATED,
                    resource_type=AuditResourceType.WATCHLIST_ENTRY,
                    resource_id=created.id,
                    request_id=request_id(http_request),
                    after=_snapshot(created),
                )
            )
            return created
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/watchlists", response_model=WatchlistListPublic)
    async def list_watchlists(
        _principal: Principal = Depends(read_access),
        list_type: Annotated[WatchlistType | None, Query(alias="listType")] = None,
        enabled: bool | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> WatchlistListPublic:
        try:
            entries = await services.watchlists.list(list_type, enabled, limit)
            return WatchlistListPublic(
                items=[WatchlistPublic.from_domain(item) for item in entries]
            )
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/watchlists/{entry_id}", response_model=WatchlistPublic)
    async def get_watchlist(
        entry_id: str,
        _principal: Principal = Depends(read_access),
    ) -> WatchlistPublic:
        try:
            return WatchlistPublic.from_domain(await services.watchlists.get(entry_id))
        except Exception as exc:
            _raise_policy_http(exc)

    @router.put(
        "/watchlists/{entry_id}",
        response_model=WatchlistPublic,
        dependencies=mutation_dependencies,
    )
    async def update_watchlist(
        entry_id: str,
        http_request: Request,
        request: WatchlistUpdateRequest,
        principal: Principal = Depends(policy_access),
    ) -> WatchlistPublic:
        try:
            before = WatchlistPublic.from_domain(await services.watchlists.get(entry_id))
            updated = WatchlistPublic.from_domain(
                await services.watchlists.update(entry_id, request.to_command())
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.WATCHLIST_UPDATED,
                    resource_type=AuditResourceType.WATCHLIST_ENTRY,
                    resource_id=entry_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_policy_http(exc)

    @router.delete(
        "/watchlists/{entry_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=mutation_dependencies,
    )
    async def delete_watchlist(
        entry_id: str,
        http_request: Request,
        principal: Principal = Depends(policy_access),
    ) -> Response:
        try:
            before = WatchlistPublic.from_domain(await services.watchlists.get(entry_id))
            await services.watchlists.delete(entry_id)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.WATCHLIST_DELETED,
                    resource_type=AuditResourceType.WATCHLIST_ENTRY,
                    resource_id=entry_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                )
            )
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as exc:
            _raise_policy_http(exc)

    @router.post(
        "/rules",
        response_model=RulePublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=mutation_dependencies,
    )
    async def create_rule(
        http_request: Request,
        request: RuleCreateRequest,
        principal: Principal = Depends(policy_access),
    ) -> RulePublic:
        try:
            created = RulePublic.from_domain(await services.rules.create(request.to_command()))
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.RULE_CREATED,
                    resource_type=AuditResourceType.RULE,
                    resource_id=created.id,
                    request_id=request_id(http_request),
                    after=_snapshot(created),
                )
            )
            return created
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/rules", response_model=RuleListPublic)
    async def list_rules(
        _principal: Principal = Depends(read_access),
        enabled_only: Annotated[bool, Query(alias="enabledOnly")] = False,
        limit: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> RuleListPublic:
        try:
            rules = await services.rules.list(enabled_only, limit)
            return RuleListPublic(items=[RulePublic.from_domain(item) for item in rules])
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/rules/{rule_id}", response_model=RulePublic)
    async def get_rule(
        rule_id: str,
        _principal: Principal = Depends(read_access),
    ) -> RulePublic:
        try:
            return RulePublic.from_domain(await services.rules.get(rule_id))
        except Exception as exc:
            _raise_policy_http(exc)

    @router.put(
        "/rules/{rule_id}",
        response_model=RulePublic,
        dependencies=mutation_dependencies,
    )
    async def update_rule(
        rule_id: str,
        http_request: Request,
        request: RuleUpdateRequest,
        principal: Principal = Depends(policy_access),
    ) -> RulePublic:
        try:
            before = RulePublic.from_domain(await services.rules.get(rule_id))
            updated = RulePublic.from_domain(
                await services.rules.update(rule_id, request.to_command())
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.RULE_UPDATED,
                    resource_type=AuditResourceType.RULE,
                    resource_id=rule_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_policy_http(exc)

    @router.delete(
        "/rules/{rule_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=mutation_dependencies,
    )
    async def delete_rule(
        rule_id: str,
        http_request: Request,
        principal: Principal = Depends(policy_access),
    ) -> Response:
        try:
            before = RulePublic.from_domain(await services.rules.get(rule_id))
            await services.rules.delete(rule_id)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.RULE_DELETED,
                    resource_type=AuditResourceType.RULE,
                    resource_id=rule_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                )
            )
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/alerts", response_model=AlertListPublic)
    async def list_alerts(
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        alert_status: Annotated[AlertStatus | None, Query(alias="status")] = None,
        plate: str | None = None,
        camera_id: Annotated[str | None, Query(alias="cameraId")] = None,
        rule_id: Annotated[str | None, Query(alias="ruleId")] = None,
    ) -> AlertListPublic:
        try:
            page = await services.alerts.list(
                AlertQuery(
                    limit=limit,
                    cursor=cursor,
                    status=alert_status,
                    plate=plate,
                    camera_id=camera_id,
                    rule_id=rule_id,
                )
            )
            return AlertListPublic(
                items=[AlertPublic.from_domain(item) for item in page.items],
                nextCursor=page.next_cursor,
            )
        except Exception as exc:
            _raise_policy_http(exc)

    @router.get("/alerts/{alert_id}", response_model=AlertPublic)
    async def get_alert(
        alert_id: str,
        _principal: Principal = Depends(read_access),
    ) -> AlertPublic:
        try:
            return AlertPublic.from_domain(await services.alerts.get(alert_id))
        except Exception as exc:
            _raise_policy_http(exc)

    @router.post(
        "/alerts/{alert_id}/acknowledge",
        response_model=AlertPublic,
        dependencies=mutation_dependencies,
    )
    async def acknowledge_alert(
        alert_id: str,
        http_request: Request,
        request: AlertTransitionRequest,
        principal: Principal = Depends(alert_access),
    ) -> AlertPublic:
        try:
            _validate_actor(request.actor_id, principal)
            before = AlertPublic.from_domain(await services.alerts.get(alert_id))
            updated = AlertPublic.from_domain(
                await services.alerts.acknowledge(alert_id, principal.id)
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.ALERT_ACKNOWLEDGED,
                    resource_type=AuditResourceType.ALERT,
                    resource_id=alert_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_policy_http(exc)

    @router.post(
        "/alerts/{alert_id}/resolve",
        response_model=AlertPublic,
        dependencies=mutation_dependencies,
    )
    async def resolve_alert(
        alert_id: str,
        http_request: Request,
        request: AlertTransitionRequest,
        principal: Principal = Depends(alert_access),
    ) -> AlertPublic:
        try:
            _validate_actor(request.actor_id, principal)
            before = AlertPublic.from_domain(await services.alerts.get(alert_id))
            updated = AlertPublic.from_domain(
                await services.alerts.resolve(alert_id, principal.id)
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.ALERT_RESOLVED,
                    resource_type=AuditResourceType.ALERT,
                    resource_id=alert_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_policy_http(exc)

    return router


def _raise_policy_http(exc: Exception) -> NoReturn:
    if isinstance(exc, AuditWriteError):
        raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc
    if isinstance(exc, InvalidCursorError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, PolicyNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, PolicyConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, (RuleValidationError, ValueError)):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(status_code=503, detail="policy persistence is unavailable") from exc
    raise exc


def _snapshot(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="python", by_alias=True)


def _validate_actor(actor_id: str | None, principal: Principal) -> None:
    if actor_id is not None and actor_id != principal.id:
        raise HTTPException(
            status_code=403,
            detail="actorId must match the authenticated principal",
        )
