"""RBAC/audited camera-topology and bounded candidate routes."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, NoReturn

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.application.topology import (
    CameraTopologyService,
    CrossCameraCandidateGenerator,
)
from vehicle_intelligence.domain import AuditAction, AuditResourceType, Principal
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    CameraNotFoundError,
    PersistenceError,
    TopologyConflictError,
    TopologyNotFoundError,
)
from vehicle_intelligence.interfaces.request_context import request_id
from vehicle_intelligence.interfaces.security import APISecurity
from vehicle_intelligence.interfaces.topology_schemas import (
    CandidateListPublic,
    CandidatePublic,
    TopologyCreateRequest,
    TopologyListPublic,
    TopologyPublic,
    TopologyUpdateRequest,
)


def build_topology_router(
    service: CameraTopologyService,
    candidates: CrossCameraCandidateGenerator,
    security: APISecurity,
    audits: AuditService,
    cameras: CameraService | None = None,
    mutation_transaction: Callable[[], AsyncIterator[None]] | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["camera-topology"])
    read_access = security.require(Permission.READ_PLATFORM)
    manage_access = security.require(Permission.MANAGE_TOPOLOGY)
    mutation_dependencies = (
        [Depends(mutation_transaction)] if mutation_transaction is not None else []
    )

    async def verify_cameras(from_camera_id: str, to_camera_id: str) -> None:
        if cameras is None:
            return
        await cameras.get(from_camera_id)
        await cameras.get(to_camera_id)

    @router.post(
        "/camera-topology",
        response_model=TopologyPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=mutation_dependencies,
    )
    async def create_edge(
        http_request: Request,
        request: TopologyCreateRequest,
        principal: Principal = Depends(manage_access),
    ) -> TopologyPublic:
        try:
            await verify_cameras(request.from_camera_id, request.to_camera_id)
            created = TopologyPublic.from_domain(await service.create(request.to_command()))
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_TOPOLOGY_CREATED,
                    resource_type=AuditResourceType.CAMERA_TOPOLOGY,
                    resource_id=created.id,
                    request_id=request_id(http_request),
                    after=_snapshot(created),
                )
            )
            return created
        except Exception as exc:
            _raise_topology_http(exc)

    @router.get("/camera-topology", response_model=TopologyListPublic)
    async def list_edges(
        _principal: Principal = Depends(read_access),
        from_camera_id: Annotated[str | None, Query(alias="fromCameraId")] = None,
        to_camera_id: Annotated[str | None, Query(alias="toCameraId")] = None,
        enabled_only: Annotated[bool, Query(alias="enabledOnly")] = False,
        limit: Annotated[int, Query(ge=1, le=1000)] = 200,
    ) -> TopologyListPublic:
        try:
            edges = await service.list(
                from_camera_id=from_camera_id,
                to_camera_id=to_camera_id,
                enabled_only=enabled_only,
                limit=limit,
            )
            return TopologyListPublic(
                items=[TopologyPublic.from_domain(edge) for edge in edges]
            )
        except Exception as exc:
            _raise_topology_http(exc)

    @router.get("/camera-topology/{edge_id}", response_model=TopologyPublic)
    async def get_edge(
        edge_id: str,
        _principal: Principal = Depends(read_access),
    ) -> TopologyPublic:
        try:
            return TopologyPublic.from_domain(await service.get(edge_id))
        except Exception as exc:
            _raise_topology_http(exc)

    @router.put(
        "/camera-topology/{edge_id}",
        response_model=TopologyPublic,
        dependencies=mutation_dependencies,
    )
    async def update_edge(
        edge_id: str,
        http_request: Request,
        request: TopologyUpdateRequest,
        principal: Principal = Depends(manage_access),
    ) -> TopologyPublic:
        try:
            await verify_cameras(request.from_camera_id, request.to_camera_id)
            before = TopologyPublic.from_domain(await service.get(edge_id))
            updated = TopologyPublic.from_domain(
                await service.update(edge_id, request.to_command())
            )
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_TOPOLOGY_UPDATED,
                    resource_type=AuditResourceType.CAMERA_TOPOLOGY,
                    resource_id=edge_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                    after=_snapshot(updated),
                )
            )
            return updated
        except Exception as exc:
            _raise_topology_http(exc)

    @router.delete(
        "/camera-topology/{edge_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=mutation_dependencies,
    )
    async def delete_edge(
        edge_id: str,
        http_request: Request,
        principal: Principal = Depends(manage_access),
    ) -> Response:
        try:
            before = TopologyPublic.from_domain(await service.get(edge_id))
            await service.delete(edge_id)
            await audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_TOPOLOGY_DELETED,
                    resource_type=AuditResourceType.CAMERA_TOPOLOGY,
                    resource_id=edge_id,
                    request_id=request_id(http_request),
                    before=_snapshot(before),
                )
            )
            return Response(status_code=status.HTTP_204_NO_CONTENT)
        except Exception as exc:
            _raise_topology_http(exc)

    @router.get(
        "/vehicle-fingerprints/{fingerprint_id}/candidates",
        response_model=CandidateListPublic,
    )
    async def list_candidates(
        fingerprint_id: str,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
    ) -> CandidateListPublic:
        try:
            values = await candidates.generate(fingerprint_id, limit)
            return CandidateListPublic(
                sourceFingerprintId=fingerprint_id,
                items=[CandidatePublic.from_domain(item) for item in values],
            )
        except Exception as exc:
            _raise_topology_http(exc)

    return router


def _snapshot(model: BaseModel) -> dict[str, object]:
    return model.model_dump(mode="json", by_alias=True)


def _raise_topology_http(exc: Exception) -> NoReturn:
    if isinstance(exc, AuditWriteError):
        raise HTTPException(status_code=503, detail="audit persistence is unavailable") from exc
    if isinstance(exc, (TopologyNotFoundError, CameraNotFoundError)):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if isinstance(exc, TopologyConflictError):
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if isinstance(exc, ValueError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, PersistenceError):
        raise HTTPException(status_code=503, detail="topology persistence unavailable") from exc
    raise exc
