"""Runtime dependency probes and public health endpoints."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.dataset_registry import DatasetRegistryService
from vehicle_intelligence.application.dataset_review import DetectorDatasetReviewService
from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.application.live_monitor import (
    LiveMonitorService,
    LiveMonitorSourceState,
)
from vehicle_intelligence.application.media_access import VehicleEventMediaService
from vehicle_intelligence.application.model_training import ModelTrainingService
from vehicle_intelligence.application.realtime import RealtimeEventService, RealtimeSourceState
from vehicle_intelligence.application.runtime_health import (
    RuntimeDependency,
    RuntimeHealthService,
)
from vehicle_intelligence.config import Settings
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage
from vehicle_intelligence.interfaces.security import APISecurity


def build_runtime_health(
    settings: Settings,
    mongo: MongoRuntime | None,
    minio: MinioMediaStorage | None,
    cameras: CameraService | None,
    realtime: RealtimeEventService | None,
    live_monitor: LiveMonitorService | None,
    configured: RuntimeHealthService | None = None,
) -> RuntimeHealthService:
    """Compose health probes while preserving an explicitly injected service."""

    if configured is not None:
        return configured
    return RuntimeHealthService(
        (
            RuntimeDependency(
                "eventStore",
                required=True,
                probe=lambda: _probe_event_store(settings.mongodb.enabled, mongo),
            ),
            RuntimeDependency(
                "cameraManagement",
                required=False,
                probe=lambda: _probe_available(cameras is not None),
            ),
            RuntimeDependency(
                "minio",
                required=False,
                probe=(lambda: _probe_minio(minio))
                if settings.storage.backend == "minio"
                else None,
            ),
            RuntimeDependency(
                "realtime",
                required=False,
                probe=(lambda: _probe_realtime(realtime)) if settings.realtime.enabled else None,
            ),
            RuntimeDependency(
                "liveMonitor",
                required=False,
                probe=(lambda: _probe_live_monitor(live_monitor))
                if settings.live_monitor.enabled
                else None,
            ),
        )
    )


def build_runtime_health_router(
    runtime_health: RuntimeHealthService,
    security: APISecurity,
    cameras: CameraService | None,
    discovery: OnvifDiscoveryService | None,
    media_access: VehicleEventMediaService | None,
    detector_reviews: DetectorDatasetReviewService | None,
    dataset_registry: DatasetRegistryService | None,
    model_training: ModelTrainingService | None,
    live_monitor: LiveMonitorService | None,
    realtime: RealtimeEventService | None,
) -> APIRouter:
    """Build liveness, readiness, and capability-discovery routes."""

    router = APIRouter(tags=["system"])

    @router.get("/livez", include_in_schema=False)
    async def liveness() -> JSONResponse:
        return JSONResponse(
            content={"status": "alive"},
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/readyz", include_in_schema=False)
    async def readiness() -> JSONResponse:
        snapshot = await runtime_health.assess()
        return JSONResponse(
            status_code=(
                status.HTTP_200_OK if snapshot.ready else status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            content=snapshot.to_document(),
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/api/system/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "phase": "4",
            "authentication": "enabled" if security.enabled else "disabled",
            "cameraManagement": "available" if cameras is not None else "unavailable",
            "onvifDiscovery": "available" if discovery is not None else "disabled",
            "policyEngine": "available",
            "auditLog": "available",
            "mediaAccess": "available" if media_access is not None else "unavailable",
            "humanReview": "available",
            "datasetReview": "available" if detector_reviews is not None else "disabled",
            "datasetRegistry": "available" if dataset_registry is not None else "disabled",
            "modelTraining": "available" if model_training is not None else "disabled",
            "modelQuality": "available",
            "liveMonitor": (
                live_monitor.stats.source_state.value if live_monitor is not None else "DISABLED"
            ),
            "realtime": realtime.stats.source_state.value if realtime is not None else "DISABLED",
        }

    return router


async def _probe_event_store(mongodb_enabled: bool, mongo: MongoRuntime | None) -> bool:
    if not mongodb_enabled:
        return True
    if mongo is None:
        return False
    await mongo.ping()
    return True


async def _probe_minio(minio: MinioMediaStorage | None) -> bool:
    if minio is None:
        return False
    await minio.ping()
    return True


async def _probe_available(available: bool) -> bool:
    return available


async def _probe_realtime(service: RealtimeEventService | None) -> bool:
    return service is not None and service.stats.source_state is RealtimeSourceState.ONLINE


async def _probe_live_monitor(service: LiveMonitorService | None) -> bool:
    return service is not None and service.stats.source_state is LiveMonitorSourceState.ONLINE
