"""FastAPI event, camera, watchlist, rule, and alert management surface."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from opentelemetry import trace
from pydantic import BaseModel

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.cameras import CameraService
from vehicle_intelligence.application.dataset_registry import DatasetRegistryService
from vehicle_intelligence.application.dataset_review import DetectorDatasetReviewService
from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.application.journeys import VehicleJourneyService
from vehicle_intelligence.application.live_monitor import (
    LiveMonitorService,
    LiveMonitorSourceState,
)
from vehicle_intelligence.application.media_access import VehicleEventMediaService
from vehicle_intelligence.application.model_quality import ModelQualityService
from vehicle_intelligence.application.model_training import ModelTrainingService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.policies import (
    AlertService,
    PolicyServices,
    RuleService,
    WatchlistService,
)
from vehicle_intelligence.application.ports import (
    Authenticator,
    CameraTopologyRepository,
    DatasetSampleRepository,
    EventQuery,
    MediaStorage,
    MediaUrlSigner,
    VectorRepository,
    VehicleEventRepository,
    VehicleIdentityRepository,
)
from vehicle_intelligence.application.realtime import RealtimeEventService, RealtimeSourceState
from vehicle_intelligence.application.reid import IdentityReviewService, ReIDScoringService
from vehicle_intelligence.application.review import HumanPlateReviewService
from vehicle_intelligence.application.rules import RuleEvaluator
from vehicle_intelligence.application.runtime_health import (
    RuntimeDependency,
    RuntimeHealthService,
)
from vehicle_intelligence.application.security import (
    DevelopmentAuthenticator,
    Permission,
    StaticApiKeyAuthenticator,
)
from vehicle_intelligence.application.topology import (
    CameraTopologyService,
    CrossCameraCandidateGenerator,
)
from vehicle_intelligence.config import Settings, load_settings
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    Principal,
)
from vehicle_intelligence.exceptions import (
    AuditWriteError,
    CameraConflictError,
    CameraDiscoveryError,
    CameraNotFoundError,
    ConfigurationError,
    CredentialEncryptionError,
    PersistenceError,
)
from vehicle_intelligence.infrastructure.identity_serialization import (
    fingerprint_to_jsonable,
    identity_to_jsonable,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.live_codec import JsonLiveFrameCodec
from vehicle_intelligence.infrastructure.messaging.live_redis import RedisLiveFrameSubscriber
from vehicle_intelligence.infrastructure.messaging.realtime_redis import (
    RedisRealtimeEventSubscriber,
)
from vehicle_intelligence.infrastructure.observability.metrics import PrometheusMetrics
from vehicle_intelligence.infrastructure.observability.tracing import (
    TracingRuntime,
    build_tracing_runtime,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.persistence.audit_mongo import MongoAuditLogRepository
from vehicle_intelligence.infrastructure.persistence.camera_memory import (
    InMemoryCameraHealthRepository,
    InMemoryCameraRepository,
)
from vehicle_intelligence.infrastructure.persistence.camera_mongo import (
    MongoCameraHealthRepository,
    MongoCameraRepository,
)
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVectorRepository,
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.identity_mongo import (
    MongoVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.jsonl import JsonlVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.policy_memory import (
    InMemoryAlertRepository,
    InMemoryRuleRepository,
    InMemoryWatchlistRepository,
)
from vehicle_intelligence.infrastructure.persistence.policy_mongo import (
    MongoAlertRepository,
    MongoRuleRepository,
    MongoWatchlistRepository,
)
from vehicle_intelligence.infrastructure.persistence.quality_memory import (
    InMemoryModelQualityRepository,
)
from vehicle_intelligence.infrastructure.persistence.quality_mongo import (
    MongoModelQualityRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_mongo import (
    MongoDatasetSampleRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_memory import (
    InMemoryCameraTopologyRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_mongo import (
    MongoCameraTopologyRepository,
)
from vehicle_intelligence.infrastructure.persistence.vector_mongo import MongoVectorRepository
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher
from vehicle_intelligence.infrastructure.security.oidc import OIDCAuthenticator
from vehicle_intelligence.infrastructure.serialization import event_to_jsonable
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage
from vehicle_intelligence.infrastructure.training.dataset_registry_files import (
    FileDatasetRegistryRepository,
)
from vehicle_intelligence.infrastructure.training.dataset_review_files import (
    FileDetectorReviewRepository,
)
from vehicle_intelligence.infrastructure.training.huggingface_jobs import (
    HuggingFaceTrainingJobGateway,
)
from vehicle_intelligence.infrastructure.training.model_training_files import (
    FileModelTrainingRunRepository,
)
from vehicle_intelligence.infrastructure.vision.connection_test import (
    OpenCVCameraConnectionTester,
)
from vehicle_intelligence.infrastructure.vision.onvif_discovery import (
    WSDiscoveryOnvifProvider,
)
from vehicle_intelligence.interfaces.audit_api import build_audit_router
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
from vehicle_intelligence.interfaces.dataset_registry_api import build_dataset_registry_router
from vehicle_intelligence.interfaces.dataset_review_api import build_dataset_review_router
from vehicle_intelligence.interfaces.journey_api import build_journey_router
from vehicle_intelligence.interfaces.live_monitor_api import build_live_monitor_router
from vehicle_intelligence.interfaces.media_api import build_media_router
from vehicle_intelligence.interfaces.model_training_api import build_model_training_router
from vehicle_intelligence.interfaces.policy_api import build_policy_router
from vehicle_intelligence.interfaces.quality_api import build_quality_router
from vehicle_intelligence.interfaces.realtime_api import build_realtime_router
from vehicle_intelligence.interfaces.reid_api import build_reid_router
from vehicle_intelligence.interfaces.request_context import request_id, resolve_request_id
from vehicle_intelligence.interfaces.review_api import build_review_router
from vehicle_intelligence.interfaces.security import APISecurity, build_auth_router
from vehicle_intelligence.interfaces.topology_api import build_topology_router
from vehicle_intelligence.training.config import load_training_settings

logger = logging.getLogger(__name__)


def _repository(
    settings: Settings,
    mongo: MongoRuntime | None = None,
) -> VehicleEventRepository:
    if settings.mongodb.enabled:
        return MongoVehicleEventRepository(mongo or settings.mongodb)
    path = Path(settings.storage.output_directory) / "events.jsonl"
    return JsonlVehicleEventRepository(path)


def _camera_service(
    settings: Settings,
    mongo: MongoRuntime | None = None,
) -> CameraService | None:
    tester = OpenCVCameraConnectionTester(
        settings.rtsp,
        maximum_concurrency=settings.camera_manager.connection_test_concurrency,
    )
    if not settings.mongodb.enabled:
        return CameraService(
            InMemoryCameraRepository(),
            InMemoryCameraHealthRepository(),
            tester,
            maximum_cameras=settings.camera_manager.maximum_configured_cameras,
            batch_create_limit=settings.camera_manager.batch_create_limit,
        )
    if (
        settings.security.camera_credential_key is None
        and not settings.security.camera_credential_keys
    ):
        return None
    try:
        cipher = AesGcmCredentialCipher.from_config(settings.security)
    except ConfigurationError:
        logger.error("camera credential key is invalid; camera management disabled")
        return None
    return CameraService(
        MongoCameraRepository(mongo or settings.mongodb, cipher),
        MongoCameraHealthRepository(mongo or settings.mongodb),
        tester,
        maximum_cameras=settings.camera_manager.maximum_configured_cameras,
        batch_create_limit=settings.camera_manager.batch_create_limit,
    )


def _onvif_discovery_service(settings: Settings) -> OnvifDiscoveryService | None:
    if not settings.onvif_discovery.enabled:
        return None
    return OnvifDiscoveryService(
        WSDiscoveryOnvifProvider(settings.onvif_discovery),
        settings.onvif_discovery.maximum_results,
    )


def _policy_services(
    settings: Settings,
    normalizer: VietnamPlateNormalizer,
    mongo: MongoRuntime | None = None,
) -> PolicyServices:
    if settings.mongodb.enabled:
        source = mongo or settings.mongodb
        watchlists = MongoWatchlistRepository(source)
        rules = MongoRuleRepository(source)
        alerts = MongoAlertRepository(source)
    else:
        watchlists = InMemoryWatchlistRepository()
        rules = InMemoryRuleRepository()
        alerts = InMemoryAlertRepository()
    evaluator = RuleEvaluator()
    return PolicyServices(
        watchlists=WatchlistService(watchlists, normalizer),
        rules=RuleService(rules, evaluator),
        alerts=AlertService(alerts, normalizer),
    )


def _audit_service(settings: Settings, mongo: MongoRuntime | None = None) -> AuditService:
    repository = (
        MongoAuditLogRepository(mongo or settings.mongodb)
        if settings.mongodb.enabled
        else InMemoryAuditLogRepository()
    )
    return AuditService(repository)


def _api_security(
    settings: Settings,
    authenticator: Authenticator | None,
) -> APISecurity:
    provider = authenticator
    if provider is None:
        if not settings.auth.enabled:
            if settings.app.environment.strip().casefold() == "production":
                raise ConfigurationError("API authentication cannot be disabled in production")
            provider = DevelopmentAuthenticator()
        elif settings.auth.provider == "oidc":
            if settings.auth.oidc is None:
                raise ConfigurationError("OIDC authentication configuration is missing")
            provider = OIDCAuthenticator(settings.auth.oidc)
        else:
            provider = StaticApiKeyAuthenticator(settings.auth)
    return APISecurity(settings.auth, provider)


def _realtime_service(settings: Settings) -> RealtimeEventService | None:
    if not settings.realtime.enabled:
        return None
    codec = JsonEventEnvelopeCodec()
    source = RedisRealtimeEventSubscriber(settings.redis, settings.realtime, codec)
    return RealtimeEventService(settings.realtime, source)


def _live_monitor_service(settings: Settings) -> LiveMonitorService | None:
    if not settings.live_monitor.enabled:
        return None
    codec = JsonLiveFrameCodec(settings.live_monitor.maximum_payload_bytes)
    source = RedisLiveFrameSubscriber(settings.redis, settings.live_monitor, codec)
    return LiveMonitorService(settings.live_monitor, source)


def _media_access_service(
    settings: Settings,
    repository: VehicleEventRepository,
    signer: MediaUrlSigner | None = None,
) -> VehicleEventMediaService | None:
    if settings.storage.backend != "minio":
        return None
    return VehicleEventMediaService(
        repository,
        signer or MinioMediaStorage(settings.minio),
        settings.minio.presigned_url_ttl_seconds,
    )


def _dataset_repository(
    settings: Settings,
    repository: VehicleEventRepository,
    mongo: MongoRuntime | None = None,
) -> DatasetSampleRepository:
    return (
        MongoDatasetSampleRepository(mongo or settings.mongodb)
        if settings.mongodb.enabled and isinstance(repository, MongoVehicleEventRepository)
        else InMemoryDatasetSampleRepository()
    )


def _human_review_service(
    settings: Settings,
    repository: VehicleEventRepository,
    samples: DatasetSampleRepository,
    normalizer: VietnamPlateNormalizer,
    media: MediaStorage | None = None,
) -> HumanPlateReviewService:
    managed_media = media or (
        MinioMediaStorage(settings.minio)
        if settings.storage.backend == "minio"
        else LocalMediaStorage(settings.storage.output_directory)
    )
    return HumanPlateReviewService(repository, samples, normalizer, managed_media)


def _model_quality_service(
    settings: Settings,
    repository: VehicleEventRepository,
    samples: DatasetSampleRepository,
    mongo: MongoRuntime | None = None,
) -> ModelQualityService:
    quality_repository = (
        MongoModelQualityRepository(mongo or settings.mongodb)
        if settings.mongodb.enabled and isinstance(repository, MongoVehicleEventRepository)
        else InMemoryModelQualityRepository(
            repository,
            samples,
            settings.model_quality.in_memory_scan_limit,
        )
    )
    return ModelQualityService(quality_repository, settings.model_quality)


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


def _configured_minio(settings: Settings) -> MinioMediaStorage | None:
    if settings.storage.backend != "minio":
        return None
    return MinioMediaStorage(settings.minio)


def _build_runtime_health(
    settings: Settings,
    mongo: MongoRuntime | None,
    minio: MinioMediaStorage | None,
    cameras: CameraService | None,
    realtime: RealtimeEventService | None,
    live_monitor: LiveMonitorService | None,
    configured: RuntimeHealthService | None = None,
) -> RuntimeHealthService:
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


def _register_runtime_health_routes(
    app: FastAPI,
    runtime_health: RuntimeHealthService,
    api_security: APISecurity,
    cameras: CameraService | None,
    onvif_discovery: OnvifDiscoveryService | None,
    media_access: VehicleEventMediaService | None,
    detector_reviews: DetectorDatasetReviewService | None,
    dataset_registry: DatasetRegistryService | None,
    model_training: ModelTrainingService | None,
    live_monitor: LiveMonitorService | None,
    realtime: RealtimeEventService | None,
) -> None:
    @app.get("/livez", include_in_schema=False)
    async def liveness() -> JSONResponse:
        return JSONResponse(
            content={"status": "alive"},
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/readyz", include_in_schema=False)
    async def readiness() -> JSONResponse:
        snapshot = await runtime_health.assess()
        return JSONResponse(
            status_code=status.HTTP_200_OK
            if snapshot.ready
            else status.HTTP_503_SERVICE_UNAVAILABLE,
            content=snapshot.to_document(),
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/system/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "phase": "4",
            "authentication": "enabled" if api_security.enabled else "disabled",
            "cameraManagement": "available" if cameras is not None else "unavailable",
            "onvifDiscovery": "available" if onvif_discovery is not None else "disabled",
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


def _register_camera_health_routes(
    app: FastAPI,
    camera_service: CameraService | None,
    read_access: Callable[..., Awaitable[Principal]],
) -> None:
    @app.get("/api/camera-health", response_model=CameraHealthSnapshotPublic)
    async def list_camera_health(
        _principal: Principal = Depends(read_access),
    ) -> CameraHealthSnapshotPublic:
        service = _require_camera_service(camera_service)
        try:
            cameras, health_items = await asyncio.gather(service.list(), service.list_health())
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

    @app.get("/api/cameras/{camera_id}/health", response_model=CameraHealthPublic)
    async def get_camera_health(
        camera_id: str,
        _principal: Principal = Depends(read_access),
    ) -> CameraHealthPublic:
        service = _require_camera_service(camera_service)
        try:
            camera_health = await service.get_health(camera_id)
            if camera_health is None:
                raise HTTPException(status_code=404, detail="camera health is not available")
            return CameraHealthPublic.from_domain(camera_health)
        except HTTPException:
            raise
        except Exception as exc:
            _raise_camera_http(exc)


def create_app(
    settings: Settings | None = None,
    repository: VehicleEventRepository | None = None,
    camera_service: CameraService | None = None,
    policy_services: PolicyServices | None = None,
    audit_service: AuditService | None = None,
    authenticator: Authenticator | None = None,
    realtime_service: RealtimeEventService | None = None,
    media_access_service: VehicleEventMediaService | None = None,
    human_review_service: HumanPlateReviewService | None = None,
    live_monitor_service: LiveMonitorService | None = None,
    onvif_discovery_service: OnvifDiscoveryService | None = None,
    prometheus_metrics: PrometheusMetrics | None = None,
    tracing_runtime: TracingRuntime | None = None,
    mongo_runtime: MongoRuntime | None = None,
    vehicle_identity_repository: VehicleIdentityRepository | None = None,
    topology_repository: CameraTopologyRepository | None = None,
    vector_repository: VectorRepository | None = None,
    model_quality_service: ModelQualityService | None = None,
    detector_review_service: DetectorDatasetReviewService | None = None,
    dataset_registry_service: DatasetRegistryService | None = None,
    model_training_service: ModelTrainingService | None = None,
    runtime_health_service: RuntimeHealthService | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    camera_credentials_configured = settings.security.camera_credential_key is not None or bool(
        settings.security.camera_credential_keys
    )
    valid_camera_credentials = False
    if camera_credentials_configured:
        try:
            AesGcmCredentialCipher.from_config(settings.security)
            valid_camera_credentials = True
        except ConfigurationError:
            pass
    requires_composed_mongo = settings.mongodb.enabled and (
        repository is None
        or policy_services is None
        or audit_service is None
        or (camera_service is None and valid_camera_credentials)
        or (human_review_service is None and isinstance(repository, MongoVehicleEventRepository))
    )
    managed_mongo = mongo_runtime or (
        MongoRuntime(settings.mongodb) if requires_composed_mongo else None
    )
    owns_mongo_runtime = mongo_runtime is None and managed_mongo is not None
    mongo_to_close = managed_mongo if owns_mongo_runtime else None
    event_repository = repository or _repository(settings, managed_mongo)
    managed_identities = vehicle_identity_repository or (
        MongoVehicleIdentityRepository(managed_mongo)
        if settings.identity.enabled and managed_mongo is not None
        else InMemoryVehicleIdentityRepository(event_repository)
    )
    managed_cameras = camera_service or _camera_service(settings, managed_mongo)
    managed_topology_repository = topology_repository or (
        MongoCameraTopologyRepository(managed_mongo)
        if settings.identity.enabled and managed_mongo is not None
        else InMemoryCameraTopologyRepository()
    )
    managed_topology = CameraTopologyService(managed_topology_repository)
    managed_candidates = CrossCameraCandidateGenerator(
        managed_identities,
        managed_topology_repository,
        settings.identity,
    )
    managed_vectors = vector_repository or (
        MongoVectorRepository(
            managed_mongo,
            maximum_candidates=settings.identity.vector_candidate_limit,
        )
        if settings.identity.enabled and managed_mongo is not None
        else InMemoryVectorRepository()
    )
    managed_reid_scoring = ReIDScoringService(
        managed_identities,
        managed_candidates,
        managed_vectors,
        settings.identity.reid,
    )
    managed_identity_reviews = IdentityReviewService(
        managed_identities,
        managed_reid_scoring,
    )
    managed_journeys = VehicleJourneyService(
        managed_identities,
        event_repository,
        managed_topology_repository,
        settings.identity,
    )
    normalizer = VietnamPlateNormalizer()
    managed_policies = policy_services or _policy_services(settings, normalizer, managed_mongo)
    managed_audits = audit_service or _audit_service(settings, managed_mongo)
    api_security = _api_security(settings, authenticator)
    managed_realtime = realtime_service or _realtime_service(settings)
    managed_live_monitor = live_monitor_service or _live_monitor_service(settings)
    managed_onvif_discovery = onvif_discovery_service or _onvif_discovery_service(settings)
    managed_minio = _configured_minio(settings)
    managed_media_access = media_access_service or _media_access_service(
        settings,
        event_repository,
        managed_minio,
    )
    managed_samples = _dataset_repository(settings, event_repository, managed_mongo)
    managed_reviews = human_review_service or _human_review_service(
        settings,
        event_repository,
        managed_samples,
        normalizer,
        managed_minio,
    )
    managed_quality = model_quality_service or _model_quality_service(
        settings,
        event_repository,
        managed_samples,
        managed_mongo,
    )
    managed_detector_reviews = detector_review_service or (
        DetectorDatasetReviewService(FileDetectorReviewRepository(settings.dataset_review))
        if settings.dataset_review.enabled
        else None
    )
    managed_dataset_registry = dataset_registry_service or (
        DatasetRegistryService(FileDatasetRegistryRepository(settings.dataset_registry))
        if settings.dataset_registry.enabled
        else None
    )
    managed_model_training = model_training_service or (
        ModelTrainingService(
            FileModelTrainingRunRepository(settings.model_training),
            managed_dataset_registry,
            HuggingFaceTrainingJobGateway(),
            settings.model_training,
            load_training_settings(settings.model_training.training_config),
        )
        if settings.model_training.enabled and managed_dataset_registry is not None
        else None
    )
    event_codec = JsonEventEnvelopeCodec()
    managed_metrics = prometheus_metrics or PrometheusMetrics()
    managed_tracing = tracing_runtime or build_tracing_runtime(settings.observability)
    managed_runtime_health = _build_runtime_health(
        settings,
        managed_mongo,
        managed_minio,
        managed_cameras,
        managed_realtime,
        managed_live_monitor,
        runtime_health_service,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        primary_error: BaseException | None = None
        try:
            if managed_mongo is not None:
                await managed_mongo.initialize()
            await event_repository.ensure_indexes()
            await managed_identities.ensure_indexes()
            await managed_topology.initialize()
            await managed_reid_scoring.initialize()
            if managed_cameras is not None:
                await managed_cameras.initialize()
            await managed_policies.initialize()
            await managed_audits.initialize()
            await managed_reviews.initialize()
            if managed_detector_reviews is not None:
                await managed_detector_reviews.initialize()
                await managed_detector_reviews.flush_pending_audits(managed_audits)
                managed_detector_reviews.start_audit_relay(managed_audits)
            if managed_dataset_registry is not None:
                await managed_dataset_registry.initialize()
            if managed_model_training is not None:
                await managed_model_training.initialize()
            if managed_realtime is not None:
                await managed_realtime.initialize()
            if managed_live_monitor is not None:
                await managed_live_monitor.initialize()
            managed_runtime_health.start()
            yield
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            cleanup_error: BaseException | None = None
            closers = [("runtime health", managed_runtime_health.stop)]
            if managed_model_training is not None:
                closers.append(("model training", managed_model_training.close))
            closers.extend(
                (name, component.close)
                for name, component in (
                    ("dataset registry", managed_dataset_registry),
                    ("dataset review", managed_detector_reviews),
                    ("live monitor", managed_live_monitor),
                    ("realtime", managed_realtime),
                )
                if component is not None
            )
            closers.extend(
                (
                    ("model quality", managed_quality.close),
                    ("human review", managed_reviews.close),
                    ("audit", managed_audits.close),
                    ("policies", managed_policies.close),
                )
            )
            if managed_minio is not None:
                closers.append(("media storage", managed_minio.close))
            if managed_cameras is not None:
                closers.append(("cameras", managed_cameras.close))
            closers.extend(
                (
                    ("reid scoring", managed_reid_scoring.close),
                    ("topology", managed_topology.close),
                    ("identities", managed_identities.close),
                    ("events", event_repository.close),
                )
            )
            for component_name, close_component in closers:
                try:
                    await close_component()
                except BaseException as exc:
                    cleanup_error = cleanup_error or exc
                    logger.exception("application component cleanup failed: %s", component_name)
            try:
                if managed_tracing is not None:
                    managed_tracing.shutdown()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
                logger.exception("tracing cleanup failed")
            try:
                if mongo_to_close is not None:
                    await mongo_to_close.close()
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
                logger.exception("MongoDB cleanup failed")
            if cleanup_error is not None and primary_error is None:
                raise cleanup_error

    async def mutation_transaction() -> AsyncIterator[None]:
        if managed_mongo is None:
            yield
            return
        async with managed_mongo.transaction():
            yield

    app = FastAPI(
        title="Vehicle Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(build_auth_router(api_security))
    app.include_router(
        build_policy_router(
            managed_policies,
            api_security,
            managed_audits,
            mutation_transaction,
        )
    )
    app.include_router(build_audit_router(managed_audits, api_security))
    app.include_router(build_quality_router(managed_quality, api_security))
    app.include_router(build_journey_router(managed_journeys, api_security))
    app.include_router(
        build_reid_router(
            managed_reid_scoring,
            managed_identity_reviews,
            api_security,
            managed_audits,
            mutation_transaction,
        )
    )
    app.include_router(
        build_topology_router(
            managed_topology,
            managed_candidates,
            api_security,
            managed_audits,
            managed_cameras,
            mutation_transaction,
        )
    )
    app.include_router(
        build_review_router(
            managed_reviews,
            api_security,
            managed_audits,
            mutation_transaction,
        )
    )
    if managed_detector_reviews is not None:
        app.include_router(
            build_dataset_review_router(
                managed_detector_reviews,
                api_security,
                managed_audits,
            )
        )
    if managed_dataset_registry is not None:
        app.include_router(
            build_dataset_registry_router(
                managed_dataset_registry,
                api_security,
                managed_audits,
            )
        )
    if managed_model_training is not None:
        app.include_router(
            build_model_training_router(
                managed_model_training,
                api_security,
                managed_audits,
            )
        )
    app.include_router(build_media_router(managed_media_access, api_security))
    app.include_router(
        build_live_monitor_router(
            managed_live_monitor,
            managed_cameras,
            api_security,
        )
    )
    app.include_router(
        build_realtime_router(
            managed_realtime,
            settings.realtime,
            api_security,
            event_codec,
        )
    )

    @app.middleware("http")
    async def observe_http_request(
        http_request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        started = time.perf_counter()
        response_status = 500
        try:
            response = await call_next(http_request)
            response_status = response.status_code
            return response
        finally:
            if settings.observability.prometheus_enabled:
                route = http_request.scope.get("route")
                route_path = getattr(route, "path", "UNMATCHED")
                managed_metrics.observe_http(
                    http_request.method,
                    route_path,
                    response_status,
                    time.perf_counter() - started,
                )

    @app.middleware("http")
    async def attach_request_id(
        http_request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        http_request.state.request_id = resolve_request_id(http_request)
        span = trace.get_current_span()
        if span.is_recording():
            span.set_attribute("vehicle.request_id", http_request.state.request_id)
        response = await call_next(http_request)
        response.headers["X-Request-ID"] = http_request.state.request_id
        return response

    read_access = api_security.require(Permission.READ_PLATFORM)
    camera_admin_access = api_security.require(Permission.MANAGE_CAMERAS)
    camera_test_access = api_security.require(Permission.TEST_CAMERAS)

    @app.exception_handler(RequestValidationError)
    async def sanitized_validation_error(
        _request: object,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": jsonable_encoder(_sanitize_validation_errors(exc.errors()))},
        )

    _register_runtime_health_routes(
        app,
        managed_runtime_health,
        api_security,
        managed_cameras,
        managed_onvif_discovery,
        managed_media_access,
        managed_detector_reviews,
        managed_dataset_registry,
        managed_model_training,
        managed_live_monitor,
        managed_realtime,
    )

    if settings.observability.prometheus_enabled:

        @app.get(settings.observability.prometheus_path, include_in_schema=False)
        async def prometheus_metrics_endpoint() -> Response:
            camera_health = []
            if managed_cameras is not None:
                try:
                    camera_health = await managed_cameras.list_health()
                except PersistenceError:
                    managed_metrics.collection_errors.labels("camera_health").inc()
                    logger.exception("camera health metrics collection failed")
            return Response(
                content=managed_metrics.render(camera_health),
                headers={
                    "Content-Type": managed_metrics.content_type,
                    "Cache-Control": "no-store",
                },
            )

    @app.post(
        "/api/cameras",
        response_model=CameraPublic,
        status_code=status.HTTP_201_CREATED,
        dependencies=[Depends(mutation_transaction)],
    )
    async def create_camera(
        http_request: Request,
        request: CameraCreateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        service = _require_camera_service(managed_cameras)
        try:
            created = CameraPublic.from_domain(await service.create(request.to_command()))
            await managed_audits.record(
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

    @app.post(
        "/api/cameras/batch",
        response_model=CameraBatchPublic,
        dependencies=[Depends(mutation_transaction)],
    )
    async def create_camera_batch(
        http_request: Request,
        request: CameraBatchCreateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraBatchPublic:
        service = _require_camera_service(managed_cameras)
        try:
            result = await service.create_many(tuple(item.to_command() for item in request.items))
            public_result = CameraBatchPublic.from_domain(result)
            for item in public_result.items:
                if item.camera is None:
                    continue
                await managed_audits.record(
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

    @app.post("/api/cameras/discover", response_model=OnvifDiscoveryPublic)
    async def discover_onvif_cameras(
        http_request: Request,
        principal: Principal = Depends(camera_test_access),
    ) -> OnvifDiscoveryPublic:
        if managed_onvif_discovery is None:
            raise HTTPException(status_code=503, detail="ONVIF discovery is disabled")
        try:
            devices = await managed_onvif_discovery.discover()
            response = OnvifDiscoveryPublic(
                items=[OnvifDevicePublic.from_domain(item) for item in devices],
                count=len(devices),
            )
            await managed_audits.record(
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

    @app.get("/api/cameras", response_model=CameraListPublic)
    async def list_cameras(
        _principal: Principal = Depends(read_access),
        enabled_only: Annotated[bool, Query(alias="enabledOnly")] = False,
    ) -> CameraListPublic:
        service = _require_camera_service(managed_cameras)
        try:
            cameras = await service.list(enabled_only)
            return CameraListPublic(items=[CameraPublic.from_domain(item) for item in cameras])
        except Exception as exc:
            _raise_camera_http(exc)

    _register_camera_health_routes(app, managed_cameras, read_access)

    @app.get("/api/cameras/{camera_id}", response_model=CameraPublic)
    async def get_camera(
        camera_id: str,
        _principal: Principal = Depends(read_access),
    ) -> CameraPublic:
        service = _require_camera_service(managed_cameras)
        try:
            return CameraPublic.from_domain(await service.get(camera_id))
        except Exception as exc:
            _raise_camera_http(exc)

    @app.put(
        "/api/cameras/{camera_id}",
        response_model=CameraPublic,
        dependencies=[Depends(mutation_transaction)],
    )
    async def update_camera(
        camera_id: str,
        http_request: Request,
        request: CameraUpdateRequest,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        service = _require_camera_service(managed_cameras)
        try:
            before = CameraPublic.from_domain(await service.get(camera_id))
            updated = CameraPublic.from_domain(
                await service.update(camera_id, request.to_command())
            )
            await managed_audits.record(
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

    @app.delete(
        "/api/cameras/{camera_id}",
        status_code=status.HTTP_204_NO_CONTENT,
        dependencies=[Depends(mutation_transaction)],
    )
    async def delete_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> Response:
        service = _require_camera_service(managed_cameras)
        try:
            before = CameraPublic.from_domain(await service.get(camera_id))
            await service.delete(camera_id)
            await managed_audits.record(
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

    @app.post(
        "/api/cameras/{camera_id}/enable",
        response_model=CameraPublic,
        dependencies=[Depends(mutation_transaction)],
    )
    async def enable_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        service = _require_camera_service(managed_cameras)
        try:
            before = CameraPublic.from_domain(await service.get(camera_id))
            updated = CameraPublic.from_domain(await service.set_enabled(camera_id, True))
            await managed_audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_ENABLED,
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

    @app.post(
        "/api/cameras/{camera_id}/disable",
        response_model=CameraPublic,
        dependencies=[Depends(mutation_transaction)],
    )
    async def disable_camera(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_admin_access),
    ) -> CameraPublic:
        service = _require_camera_service(managed_cameras)
        try:
            before = CameraPublic.from_domain(await service.get(camera_id))
            updated = CameraPublic.from_domain(await service.set_enabled(camera_id, False))
            await managed_audits.record(
                AuditRecord(
                    principal=principal,
                    action=AuditAction.CAMERA_DISABLED,
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

    @app.post(
        "/api/cameras/{camera_id}/test-connection",
        response_model=CameraConnectionTestPublic,
    )
    async def test_camera_connection(
        camera_id: str,
        http_request: Request,
        principal: Principal = Depends(camera_test_access),
    ) -> CameraConnectionTestPublic:
        service = _require_camera_service(managed_cameras)
        try:
            camera = CameraPublic.from_domain(await service.get(camera_id))
            result = await service.test_connection(camera_id)
            public_result = CameraConnectionTestPublic.from_domain(result)
            await managed_audits.record(
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

    @app.get("/api/events")
    async def list_events(
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: str | None = None,
        camera_id: Annotated[str | None, Query(alias="cameraId")] = None,
        plate: str | None = None,
        event_type: Annotated[str | None, Query(alias="eventType")] = None,
        direction: str | None = None,
        status: str | None = None,
        from_time: Annotated[datetime | None, Query(alias="from")] = None,
        to_time: Annotated[datetime | None, Query(alias="to")] = None,
    ) -> dict[str, object]:
        canonical = _canonical_plate(normalizer, plate)
        _validate_aware(from_time, "from")
        _validate_aware(to_time, "to")
        try:
            page = await event_repository.list(
                EventQuery(
                    limit=limit,
                    cursor=cursor,
                    camera_id=camera_id,
                    plate=canonical,
                    event_type=event_type,
                    direction=direction,
                    status=status,
                    from_time=from_time,
                    to_time=to_time,
                )
            )
        except PersistenceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "items": [event_to_jsonable(event) for event in page.items],
            "nextCursor": page.next_cursor,
        }

    @app.get("/api/events/{event_id}")
    async def get_event(
        event_id: str,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        event = await event_repository.get(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="vehicle event not found")
        return event_to_jsonable(event)

    @app.get("/api/vehicles/search")
    async def search_vehicles(
        plate: Annotated[str, Query(min_length=4)],
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        cursor: str | None = None,
    ) -> dict[str, object]:
        canonical = _canonical_plate(normalizer, plate)
        if canonical is None:
            raise HTTPException(status_code=422, detail="plate is required")
        try:
            page = await event_repository.list(
                EventQuery(limit=limit, cursor=cursor, plate=canonical)
            )
        except PersistenceError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "query": canonical,
            "items": [event_to_jsonable(event) for event in page.items],
            "nextCursor": page.next_cursor,
        }

    @app.get("/api/vehicles/{vehicle_id}")
    async def get_vehicle_identity(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        identity = await managed_identities.get(vehicle_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="vehicle identity not found")
        result = identity_to_jsonable(identity)
        latest = await managed_journeys.latest(vehicle_id)
        result["latestEvent"] = event_to_jsonable(latest) if latest is not None else None
        return result

    @app.get("/api/vehicles/{vehicle_id}/fingerprints")
    async def list_vehicle_fingerprints(
        vehicle_id: str,
        _principal: Principal = Depends(read_access),
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
    ) -> dict[str, object]:
        identity = await managed_identities.get(vehicle_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="vehicle identity not found")
        fingerprints = await managed_identities.list_fingerprints(vehicle_id, limit)
        return {
            "vehicleId": vehicle_id,
            "items": [fingerprint_to_jsonable(item) for item in fingerprints],
        }

    if managed_tracing is not None:
        managed_tracing.instrument(app)
    return app


def _canonical_plate(normalizer: VietnamPlateNormalizer, plate: str | None) -> str | None:
    if plate is None:
        return None
    normalized = normalizer.normalize(plate)
    if not normalized.valid or normalized.normalized is None:
        raise HTTPException(status_code=422, detail="invalid Vietnamese plate format")
    return normalized.normalized


def _validate_aware(value: datetime | None, field: str) -> None:
    if value is not None and value.tzinfo is None:
        raise HTTPException(status_code=422, detail=f"{field} timestamp must include a timezone")


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


def _sanitize_validation_errors(
    errors: list[dict[str, object]],
) -> list[dict[str, object]]:
    sensitive = ("rtsp", "password", "secret", "credential", "token")
    sanitized: list[dict[str, object]] = []
    for error in errors:
        item = dict(error)
        raw_location = item.get("loc")
        location = raw_location if isinstance(raw_location, (list, tuple)) else ()
        if any(fragment in str(part).lower() for part in location for fragment in sensitive):
            item["input"] = "[REDACTED]"
        elif "input" in item:
            item["input"] = _redact_sensitive_input(item["input"], sensitive)
        sanitized.append(item)
    return sanitized


def _redact_sensitive_input(value: object, sensitive: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        return {
            key: (
                "[REDACTED]"
                if any(fragment in str(key).lower() for fragment in sensitive)
                else _redact_sensitive_input(item, sensitive)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_input(item, sensitive) for item in value]
    if isinstance(value, str) and value.lower().startswith(("rtsp://", "rtsps://")):
        return "[REDACTED]"
    return value


def main() -> None:
    import uvicorn

    uvicorn.run("vehicle_intelligence.interfaces.api:app", host="0.0.0.0", port=8000)


app = create_app()
