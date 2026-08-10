"""Shared vision-worker composition for file and RTSP entry points."""

from __future__ import annotations

from collections.abc import Callable

from vehicle_intelligence.application.direction import DirectionEstimator
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.health import CameraHealthReporter
from vehicle_intelligence.application.live_preview import LivePreviewReporter
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.pipeline import PipelineResult, VideoVehiclePipeline
from vehicle_intelligence.application.ports import (
    CameraHealthRepository,
    VehicleEventPublisher,
    VehicleEventRepository,
    VideoSource,
)
from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.application.selection import BestFrameSelector
from vehicle_intelligence.application.voting import PlateCandidateAggregator
from vehicle_intelligence.config import Settings
from vehicle_intelligence.domain import Direction, Point, VehicleEvent
from vehicle_intelligence.exceptions import ConfigurationError
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec
from vehicle_intelligence.infrastructure.messaging.direct import RepositoryEventPublisher
from vehicle_intelligence.infrastructure.messaging.live_codec import JsonLiveFrameCodec
from vehicle_intelligence.infrastructure.messaging.live_redis import RedisLiveFramePublisher
from vehicle_intelligence.infrastructure.messaging.redis_streams import (
    RedisStreamEventPublisher,
)
from vehicle_intelligence.infrastructure.persistence.composite import (
    CompositeVehicleEventRepository,
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


def validate_runtime_settings(settings: Settings) -> None:
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
    if settings.tracking.provider != "bytetrack":
        raise ConfigurationError(
            f"this runtime provides bytetrack for the tracker; got {settings.tracking.provider}"
        )


async def execute_pipeline(
    settings: Settings,
    source: VideoSource,
    *,
    retain_events: bool = True,
    event_observer: Callable[[VehicleEvent], None] | None = None,
    health_repository: CameraHealthRepository | None = None,
) -> PipelineResult:
    publisher = _publisher(settings)
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
        if health_reporter is not None:
            await health_reporter.initialize()
        if live_preview is not None:
            await live_preview.initialize()
        pipeline = _pipeline(
            settings,
            source,
            publisher,
            retain_events=retain_events,
            event_observer=event_observer,
            health_reporter=health_reporter,
            live_preview=live_preview,
        )
        return await pipeline.run()
    finally:
        source.close()
        try:
            if live_preview is not None:
                await live_preview.close()
        finally:
            try:
                await publisher.close()
            finally:
                if health_reporter is not None:
                    await health_reporter.close()


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
    output = settings.storage.output_directory
    jsonl = JsonlVehicleEventRepository(output / "events.jsonl")
    if not settings.mongodb.enabled:
        return jsonl
    return CompositeVehicleEventRepository((jsonl, MongoVehicleEventRepository(settings.mongodb)))


def _pipeline(
    settings: Settings,
    source: VideoSource,
    publisher: VehicleEventPublisher,
    *,
    retain_events: bool,
    event_observer: Callable[[VehicleEvent], None] | None,
    health_reporter: CameraHealthReporter | None,
    live_preview: LivePreviewReporter | None,
) -> VideoVehiclePipeline:
    media_storage = (
        MinioMediaStorage(settings.minio)
        if settings.storage.backend == "minio"
        else LocalMediaStorage(settings.storage.output_directory)
    )
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
    )
    return VideoVehiclePipeline(
        settings=settings,
        source=source,
        vehicle_detector=(
            None
            if settings.vision.plate_only
            else create_vehicle_detector(settings.vision.vehicle_detection)
        ),
        tracker=ByteTrackVehicleTracker(
            settings.tracking,
            min(source.source_fps, settings.camera.fps_limit),
        ),
        plate_detector=create_plate_detector(settings.vision.plate_detection),
        quality_evaluator=PlateQualityEvaluator(settings.vision.plate_quality),
        preprocessor=AdaptivePlatePreprocessor(settings.vision.preprocessing),
        ocr=PaddleOCRProvider(settings.vision.ocr),
        normalizer=normalizer,
        selector=BestFrameSelector(settings.vision.snapshot_selection),
        finalizer=finalizer,
        direction_estimator=direction,
        retain_events=retain_events,
        event_observer=event_observer,
        health_reporter=health_reporter,
        live_preview=live_preview,
    )
