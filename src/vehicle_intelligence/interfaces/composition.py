"""Shared vision-worker composition for file and RTSP entry points."""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.finalization_outbox import FinalizationOutbox
from vehicle_intelligence.application.health import CameraHealthReporter
from vehicle_intelligence.application.live_preview import LivePreviewReporter
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.pipeline import PipelineResult, VideoVehiclePipeline
from vehicle_intelligence.application.ports import (
    CameraHealthRepository,
    MediaStorage,
    MediaStorageLifecycle,
    PlateDetector,
    VehicleDetector,
    VehicleEventPublisher,
    VehicleEventRepository,
    VideoSource,
)
from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.application.selection import BestFrameSelector
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import Settings
from vehicle_intelligence.domain import Direction, Point, VehicleEvent
from vehicle_intelligence.exceptions import ConfigurationError, EventBusError, PersistenceError
from vehicle_intelligence.infrastructure.finalization_outbox import (
    FilesystemFinalizationOutbox,
)
from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_CAMERA_ENV,
    INFERENCE_SOCKET_ENV,
    INFERENCE_TOKEN_ENV,
    INFERENCE_TOKEN_FD_ENV,
    read_inference_token,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.messaging.live_codec import JsonLiveFrameCodec
from vehicle_intelligence.infrastructure.messaging.live_redis import RedisLiveFramePublisher
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    RedisStreamEventPublisher,
)
from vehicle_intelligence.infrastructure.persistence.jsonl import JsonlVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage
from vehicle_intelligence.infrastructure.vision.bytetrack import ByteTrackVehicleTracker
from vehicle_intelligence.infrastructure.vision.factory import (
    create_plate_detector,
    create_vehicle_detector,
    validate_detector_provider,
)
from vehicle_intelligence.infrastructure.vision.opencv import (
    AdaptivePlatePreprocessor,
    OpenCVImageEncoder,
    OpenCVLivePreviewEncoder,
)
from vehicle_intelligence.infrastructure.vision.paddleocr import PaddleOCRProvider
from vehicle_intelligence.infrastructure.vision.remote import (
    RemotePlateDetector,
    RemoteVehicleDetector,
    UnixInferenceClient,
)

logger = logging.getLogger(__name__)


def validate_runtime_settings(settings: Settings) -> None:
    _validate_camera_finalization_budget(settings)
    if not settings.vision.plate_detection.model_path:
        raise ConfigurationError(
            "a trained Vietnamese plate checkpoint is required; use --plate-model or config"
        )
    providers = [(settings.vision.plate_detection.provider, "plate detector")]
    if not settings.vision.plate_only:
        providers.insert(
            0,
            (settings.vision.vehicle_detection.provider, "vehicle detector"),
        )
    for configured, component in providers:
        validate_detector_provider(configured, component)
    if settings.vision.ocr.provider != "paddleocr":
        raise ConfigurationError(
            "this runtime provides paddleocr for the OCR provider; "
            f"got {settings.vision.ocr.provider}"
        )
    production = settings.app.environment.strip().casefold() == "production"
    if production and not all(
        (
            settings.vision.ocr.detection_model_directory,
            settings.vision.ocr.detection_model_hash,
            settings.vision.ocr.recognition_model_directory,
            settings.vision.ocr.recognition_model_hash,
        )
    ):
        raise ConfigurationError(
            "production OCR requires local detection/recognition model directories "
            "and SHA-256 manifest hashes"
        )
    if settings.tracking.provider != "bytetrack":
        raise ConfigurationError(
            f"this runtime provides bytetrack for the tracker; got {settings.tracking.provider}"
        )
    if settings.gpu_scheduler.enabled:
        expected_socket = str(settings.gpu_scheduler.socket_path)
        if os.environ.get(INFERENCE_SOCKET_ENV) != expected_socket:
            raise ConfigurationError("camera worker shared inference socket does not match config")
        if os.environ.get(INFERENCE_CAMERA_ENV) != settings.camera.id:
            raise ConfigurationError(
                "camera worker shared inference identity does not match config"
            )
        if not (os.environ.get(INFERENCE_TOKEN_FD_ENV) or os.environ.get(INFERENCE_TOKEN_ENV)):
            raise ConfigurationError("camera worker shared inference capability is missing")


async def execute_pipeline(
    settings: Settings,
    source: VideoSource,
    *,
    retain_events: bool = True,
    event_observer: Callable[[VehicleEvent], None] | None = None,
    health_repository: CameraHealthRepository | None = None,
) -> PipelineResult:
    publisher: VehicleEventPublisher | None = None
    media_storage: MediaStorage | None = None
    outbox: FinalizationOutbox | None = None
    live_preview: LivePreviewReporter | None = None
    health_reporter: CameraHealthReporter | None = None
    primary_error: BaseException | None = None
    try:
        _validate_camera_finalization_budget(settings)
        publisher = _publisher(settings)
        media_storage = _media_storage(settings)
        outbox = _finalization_outbox(settings, media_storage, publisher)
        live_preview = _live_preview(settings)
        health_reporter = (
            CameraHealthReporter(
                health_repository,
                settings.camera_manager.health_publish_interval_seconds,
            )
            if health_repository is not None
            else None
        )
        try:
            await publisher.initialize()
        except (EventBusError, PersistenceError):
            if outbox is None:
                raise
            logger.warning(
                "event publisher unavailable; durable finalization staging remains active",
                extra={"camera_id": settings.camera.id},
                exc_info=True,
            )
        if outbox is not None:
            await outbox.initialize()
        if health_reporter is not None:
            await health_reporter.initialize()
        if live_preview is not None:
            await live_preview.initialize()
        pipeline = _pipeline(
            settings,
            source,
            publisher,
            media_storage,
            outbox,
            retain_events=retain_events,
            event_observer=event_observer,
            health_reporter=health_reporter,
            live_preview=live_preview,
        )
        return await pipeline.run()
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_error = await _close_pipeline_resources(
            source,
            live_preview,
            outbox,
            publisher,
            media_storage,
            health_reporter,
            settings.camera.id,
        )
        if primary_error is None and cleanup_error is not None:
            raise cleanup_error


def _validate_camera_finalization_budget(settings: Settings) -> None:
    try:
        settings.validate_camera_finalization_budget()
    except ValueError as exc:
        raise ConfigurationError(str(exc)) from exc


async def _close_pipeline_resources(
    source: VideoSource,
    live_preview: LivePreviewReporter | None,
    outbox: FinalizationOutbox | None,
    publisher: VehicleEventPublisher | None,
    media_storage: MediaStorage | None,
    health_reporter: CameraHealthReporter | None,
    camera_id: str,
) -> BaseException | None:
    cleanup_error: BaseException | None = None
    try:
        source.close()
    except BaseException as exc:
        cleanup_error = exc
        logger.exception(
            "pipeline resource cleanup failed",
            extra={"camera_id": camera_id, "component": "source"},
        )

    closers: list[tuple[str, Callable[[], Awaitable[None]]]] = []
    if live_preview is not None:
        closers.append(("live preview", live_preview.close))
    if outbox is not None:
        closers.append(("finalization outbox", outbox.close))
    if publisher is not None:
        closers.append(("event publisher", publisher.close))
    if isinstance(media_storage, MediaStorageLifecycle):
        closers.append(("media storage", media_storage.close))
    if health_reporter is not None:
        closers.append(("camera health", health_reporter.close))

    for name, close in closers:
        try:
            await close()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error = exc
            logger.exception(
                "pipeline resource cleanup failed",
                extra={"camera_id": camera_id, "component": name},
            )
    return cleanup_error


def _publisher(settings: Settings) -> VehicleEventPublisher:
    if settings.event_bus.backend == "redis":
        return RedisStreamEventPublisher(settings.redis, JsonEventEnvelopeCodec())
    return RepositoryEventPublisher(_repository(settings))


def _live_preview(settings: Settings) -> LivePreviewReporter | None:
    if not settings.live_monitor.enabled:
        return None
    codec = JsonLiveFrameCodec(settings.live_monitor.maximum_payload_bytes)
    publisher = RedisLiveFramePublisher(settings.redis, settings.live_monitor, codec)
    return LivePreviewReporter(
        settings.live_monitor,
        OpenCVLivePreviewEncoder(),
        publisher,
    )


def _repository(settings: Settings) -> VehicleEventRepository:
    if settings.mongodb.enabled:
        return MongoVehicleEventRepository(settings.mongodb)
    output = settings.storage.output_directory
    return JsonlVehicleEventRepository(output / "events.jsonl")


def _media_storage(settings: Settings) -> MediaStorage:
    if settings.storage.backend == "minio":
        return MinioMediaStorage(settings.minio)
    return LocalMediaStorage(settings.storage.output_directory)


def _finalization_outbox(
    settings: Settings,
    media_storage: MediaStorage,
    publisher: VehicleEventPublisher,
) -> FinalizationOutbox | None:
    if not settings.finalization_outbox.enabled:
        return None
    return FilesystemFinalizationOutbox(
        settings.finalization_outbox,
        settings.storage.output_directory,
        settings.camera.id,
        JsonEventEnvelopeCodec(),
        media_storage,
        publisher,
    )


def _pipeline(
    settings: Settings,
    source: VideoSource,
    publisher: VehicleEventPublisher,
    media_storage: MediaStorage,
    outbox: FinalizationOutbox | None,
    *,
    retain_events: bool,
    event_observer: Callable[[VehicleEvent], None] | None,
    health_reporter: CameraHealthReporter | None,
    live_preview: LivePreviewReporter | None,
) -> VideoVehiclePipeline:
    vehicle_detector, plate_detector = _detectors(settings)
    line = (
        (Point(*settings.camera.crossing_line[0]), Point(*settings.camera.crossing_line[1]))
        if settings.camera.crossing_line is not None
        else None
    )
    direction = DirectionEstimator(
        line,
        Direction(settings.camera.crossing_positive_to_negative),
        settings.camera.direction,
    )
    normalizer = VietnamPlateNormalizer(
        allow_partial=settings.vision.ocr.allow_partial_plate,
        partial_min_characters=settings.vision.ocr.partial_min_characters,
        partial_max_characters=settings.vision.ocr.partial_max_characters,
    )
    finalizer = VehicleEventFinalizer(
        camera=settings.camera,
        events=settings.events,
        storage_config=settings.storage,
        aggregator=PlateCandidateAggregator(settings.voting, normalizer),
        direction_estimator=direction,
        media_storage=media_storage,
        image_encoder=OpenCVImageEncoder(),
        publisher=publisher,
        config_version=settings.app.config_version,
        outbox=outbox,
    )
    return VideoVehiclePipeline(
        settings=settings,
        source=source,
        vehicle_detector=(None if settings.vision.plate_only else vehicle_detector),
        tracker=ByteTrackVehicleTracker(
            settings.tracking,
            min(source.source_fps, settings.camera.fps_limit),
        ),
        plate_detector=plate_detector,
        quality_evaluator=PlateQualityEvaluator(settings.vision.plate_quality),
        preprocessor=AdaptivePlatePreprocessor(settings.vision.preprocessing),
        ocr=PaddleOCRProvider(
            settings.vision.ocr,
            require_local_artifacts=(settings.app.environment.strip().casefold() == "production"),
        ),
        normalizer=normalizer,
        selector=BestFrameSelector(settings.vision.snapshot_selection),
        finalizer=finalizer,
        direction_estimator=direction,
        retain_events=retain_events,
        event_observer=event_observer,
        health_reporter=health_reporter,
        live_preview=live_preview,
    )


def _detectors(settings: Settings) -> tuple[VehicleDetector | None, PlateDetector]:
    if not settings.gpu_scheduler.enabled:
        vehicle = (
            None
            if settings.vision.plate_only
            else create_vehicle_detector(settings.vision.vehicle_detection)
        )
        return vehicle, create_plate_detector(settings.vision.plate_detection)
    socket_path = os.environ.get(INFERENCE_SOCKET_ENV)
    camera_id = os.environ.get(INFERENCE_CAMERA_ENV)
    if socket_path is None or socket_path != str(settings.gpu_scheduler.socket_path):
        raise ConfigurationError("camera worker shared inference endpoint is invalid")
    if camera_id is None or camera_id != settings.camera.id:
        raise ConfigurationError("camera worker shared inference endpoint is invalid")
    client = UnixInferenceClient(
        socket_path,
        camera_id,
        read_inference_token(),
        timeout_seconds=settings.gpu_scheduler.request_timeout_seconds,
        maximum_payload_bytes=settings.gpu_scheduler.maximum_payload_bytes,
        maximum_images=settings.gpu_scheduler.maximum_images_per_request,
    )
    vehicle = None if settings.vision.plate_only else RemoteVehicleDetector(client)
    return vehicle, RemotePlateDetector(client)
