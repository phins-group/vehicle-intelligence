"""RBAC-protected camera management, discovery, and health routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.params import Depends as DependsParameter
from pydantic import BaseModel

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    CameraConflictError,
    CameraDiscoveryError,
    CameraNotFoundError,
    CredentialEncryptionError,
    PersistenceError,
)
from vehicle_intelligence.interfaces.camera_schemas import (
    CameraBatchCreateRequest,
    CameraBatchPublic,
    CameraConnectionTestPublic,
    CameraCreateRequest,
    CameraHealthPublic,
    CameraHealthSnapshotItemPublic,
    CameraHealthSnapshotPublic,
    CameraListPublic,
    CameraPublic,
    CameraUpdateRequest,
    OnvifDevicePublic,
    OnvifDiscoveryPublic,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity

AccessDependency = Callable[..., Awaitable[Principal]]
MutationTransaction = Callable[[], AsyncIterator[None]]


def build_camera_router(
    service: CameraService | None,
    discovery: OnvifDiscoveryService | None,
    security: APISecurity,
    audits: AuditService,
    mutation_transaction: MutationTransaction | None = None,
) -> APIRouter:
    """Build the complete camera surface without coupling it to app composition."""

    router = APIRouter(prefix="/api", tags=["cameras"])
    read_access = security.require(Permission.READ_PLATFORM)
    camera_admin_access = security.require(Permission.MANAGE_CAMERAS)
    camera_test_access = security.require(Permission.TEST_CAMERAS)
    mutation_dependencies = _mutation_dependencies(mutation_transaction)

    _register_camera_create_routes(
        router,
        service,
        audits,
        camera_admin_access,
        mutation_dependencies,
    )
    _register_camera_discovery_route(router, discovery, audits, camera_test_access)
    _register_camera_read_routes(router, service, read_access)
    _register_camera_mutation_routes(
        router,
        service,
        audits,
        camera_admin_access,
        mutation_dependencies,
    )
    _register_camera_connection_test_route(
        router,
        service,
        audits,
        camera_test_access,
    )
    return router


def _register_camera_create_routes(
    router: APIRouter,
    service: CameraService | None,
    audits: AuditService,
    camera_admin_access: AccessDependency,
    mutation_dependencies: Sequence[DependsParameter],
) -> None:

    @router.post(
        "/cameras",
        response_model=CameraPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=mutation_dependencies,
    )
    async def create_camera(
        http_request: Request,
        request: CameraCreateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        camera_service = _require_camera_service(service)
        try:
            created = CameraPublic.from_domain(await camera_service.create(request.to_command()))
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_CREATED,
                    resource_type=AuditResourceType.CAMERA,
                    resource_id=created.id,
                    request_id=request_id(http_request),
                    after=_snapshot(created),
                )
            )
            return created
        except Exception as exc:
            _raise_camera_http(exc)

    @router.post(
        "/cameras/batch",
        response_model=CameraBatchPublic,
        dependencies=mutation_dependencies,
    )
    async def create_camera_batch(
        http_request: Request,
        request: CameraBatchCreateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraBatchPublic:
        camera_service = _require_camera_service(service)
        try:
            result = await camera_service.create_many(
                tuple(item.to_command() for item in request.items)
            )
            public_result = CameraBatchPublic.from_domain(result)
            for item in public_result.items:
                if item.camera is None:
                    continue
                await audits.record(
                    AuditRecord(
                        principal=principal,
                        action=AuditAction.CAMERA_CREATED,
                        resource_type=AuditResourceType.CAMERA,
                        resource_id=item.camera_id,
                        request_id=request_id(http_request),
                        after=_snapshot(item.camera),
                        metadata={"source": "BATCH"},
                    )
                )
            return public_result
        except Exception as exc:
            _raise_camera_http(exc)


def _register_camera_discovery_route(
    router: APIRouter,
    discovery: OnvifDiscoveryService | None,
    audits: AuditService,
    camera_test_access: AccessDependency,
) -> None:

    @router.post("/cameras/discover", response_model=OnvifDiscoveryPublic)
    async def discover_onvif_cameras(
        http_request: Request,
        principal: Principal = Depends(camera_test_access),
    ) -> OnvifDiscoveryPublic:
        if discovery is None:
            raise HTTPException(status_code=503, detail="ONVIF discovery is disabled")
        try:
            devices = await discovery.discover()
            response = OnvifDiscoveryPublic(
                items=[OnvifDevicePublic.from_domain(item) for item in devices],
                count=len(devices),
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_DISCOVERY_RUN,
                    resource_type=AuditResourceType.CAMERA,
                    resource_id="onvif-discovery",
                    request_id=request_id(http_request),
                    metadata={"resultCount": len(devices)},
                )
            )
            return response
        except CameraDiscoveryError as exc:
            raise HTTPException(
                status_code=503,
                detail="ONVIF discovery is temporarily unavailable",
            ) from exc
        except AuditWriteError as exc:
            raise HTTPException(
                status_code=503,
                detail="audit persistence is unavailable",
            ) from exc


def _register_camera_read_routes(
    router: APIRouter,
    service: CameraService | None,
    read_access: AccessDependency,
) -> None:

    @router.get("/cameras", response_model=CameraListPublic)
    async def list_cameras(
        _principal: Principal = Depends(read_access),
        enabled_only: Annotated[bool, Query(alias="enabledOnly")] = False,
    ) -> CameraListPublic:
        camera_service = _require_camera_service(service)
        try:
            cameras = await camera_service.list(enabled_only)
            return CameraListPublic(items=[CameraPublic.from_domain(item) for item in cameras])
        except Exception as exc:
            _raise_camera_http(exc)

    @router.get("/camera-health", response_model=CameraHealthSnapshotPublic)
    async def list_camera_health(
        _principal: Principal = Depends(read_access),
    ) -> CameraHealthSnapshotPublic:
        camera_service = _require_camera_service(service)
        try:
            cameras, health_items = await asyncio.gather(
                camera_service.list(), camera_service.list_health()
            )
            health_by_camera = {item.camera_id: item for item in health_items}
            return CameraHealthSnapshotPublic(
                items=[
                    CameraHealthSnapshotItemPublic(
                        camera=CameraPublic.from_domain(camera),
                        health=(
                            CameraHealthPublic.from_domain(health)
                            if (health := health_by_camera.get(camera.id)) is not None
                            else None
                        ),
                    )
                    for camera in cameras
                ]
            )
        except Exception as exc:
            _raise_camera_http(exc)

    @router.get("/cameras/{camera_id}/health", response_model=CameraHealthPublic)
    async def get_camera_health(
        camera_id: str,
        _principal: Principal = Depends(read_access),
    ) -> CameraHealthPublic:
        camera_service = _require_camera_service(service)
        try:
            camera_health = await camera_service.get_health(camera_id)
            if camera_health is None:
                raise HTTPException(status_code=404, detail="camera health is not available")
            return CameraHealthPublic.from_domain(camera_health)
        except HTTPException:
            raise
        except Exception as exc:
            _raise_camera_http(exc)

    @router.get("/cameras/{camera_id}", response_model=CameraPublic)
    async def get_camera(
        camera_id: str,
        _principal: Principal = Depends(read_access),
    ) -> CameraPublic:
        camera_service = _require_camera_service(service)
        try:
            return CameraPublic.from_domain(await camera_service.get(camera_id))
        except Exception as exc:
            _raise_camera_http(exc)


def _register_camera_mutation_routes(
    router: APIRouter,
    service: CameraService | None,
    audits: AuditService,
    camera_admin_access: AccessDependency,
    mutation_dependencies: Sequence[DependsParameter],
) -> None:

    @router.put(
        "/cameras/{camera_id}",
        response_model=CameraPublic,
        dependencies=mutation_dependencies,
    )
    async def update_camera(
        camera_id: str,
        http_request: Request,
        request: CameraUpdateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        camera_service = _require_camera_service(service)
        try:
            before = CameraPublic.from_domain(await camera_service.get(camera_id))
            updated = CameraPublic.from_domain(
                await camera_service.update(camera_id, request.to_command())
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_UPDATED,
                    resource_type=AuditResourceType.CAMERA,
                    resource_id=camera_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_camera_http(exc)

    @router.delete(
        "/cameras/{camera_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=mutation_dependencies,
    )
    async def delete_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> Response:
        camera_service = _require_camera_service(service)
        try:
            before = CameraPublic.from_domain(await camera_service.get(camera_id))
            await camera_service.delete(camera_id)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_DELETED,
                    resource_type=AuditResourceType.CAMERA,
                    resource_id=camera_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                )
            )
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as exc:
            _raise_camera_http(exc)

    @router.post(
        "/cameras/{camera_id}/enable",
        response_model=CameraPublic,
        dependencies=mutation_dependencies,
    )
    async def enable_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        return await _set_camera_enabled(
            service,
            audits,
            camera_id,
            enabled=True,
            principal=principal,
            http_request=http_request,
        )

    @router.post(
        "/cameras/{camera_id}/disable",
        response_model=CameraPublic,
        dependencies=mutation_dependencies,
    )
    async def disable_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        return await _set_camera_enabled(
            service,
            audits,
            camera_id,
            enabled=False,
            principal=principal,
            http_request=http_request,
        )


def _register_camera_connection_test_route(
    router: APIRouter,
    service: CameraService | None,
    audits: AuditService,
    camera_test_access: AccessDependency,
) -> None:

    @router.post(
        "/cameras/{camera_id}/test-connection",
        response_model=CameraConnectionTestPublic,
    )
    async def test_camera_connection(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_test_access),
    ) -> CameraConnectionTestPublic:
        camera_service = _require_camera_service(service)
        try:
            camera = CameraPublic.from_domain(await camera_service.get(camera_id))
            result = await camera_service.test_connection(camera_id)
            public_result = CameraConnectionTestPublic.from_domain(result)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_CONNECTION_TESTED,
                    resource_type=AuditResourceType.CAMERA,
                    resource_id=camera_id,
                    request_id=request_id(http_request),
                    before=_snapshot(camera),
                    after=_snapshot(public_result),
                )
            )
            return public_result
        except Exception as exc:
            _raise_camera_http(exc)


def _mutation_dependencies(
    mutation_transaction: MutationTransaction | None,
) -> list[DependsParameter]:
    return [Depends(mutation_transaction)] if mutation_transaction is not None else []


async def _set_camera_enabled(
    service: CameraService | None,
    audits: AuditService,
    camera_id: str,
    *,
    enabled: bool,
    principal: Principal,
    http_request: Request,
) -> CameraPublic:
    camera_service = _require_camera_service(service)
    try:
        before = CameraPublic.from_domain(await camera_service.get(camera_id))
        updated = CameraPublic.from_domain(await camera_service.set_enabled(camera_id, enabled))
        await audits.record(
            AuditRecord(
                principal=principal,
                action=(AuditAction.CAMERA_ENABLED if enabled else AuditAction.CAMERA_DISABLED),
                resource_type=AuditResourceType.CAMERA,
                resource_id=camera_id,
                request_id=request_id(http_request),
                before=_snapshot(before),
                after=_snapshot(updated),
            )
        )
        return updated
    except Exception as exc:
        _raise_camera_http(exc)


def _require_camera_service(service: CameraService | None) -> CameraService:
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="camera management requires a configured credential encryption key",
        )
    return service


def _raise_camera_http(exc: Exception) -> NoReturn:
    if isinstance(exc, AuditWriteError):
        raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc
    if isinstance(exc, CameraNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, CameraConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, CredentialEncryptionError):
        raise HTTPException(status_code=503, detail="camera credential is unavailable") from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(status_code=503, detail="camera persistence is unavailable") from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    raise exc


def _snapshot(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="python", by_alias=True)
