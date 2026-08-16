import asyncio

import pytest
from fastapi.testclient import TestClient

from vehicle_intelligence.application.pipeline import PipelineResult, PipelineStats
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError, EventBusError
from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_CAMERA_ENV,
    INFERENCE_SOCKET_ENV,
    INFERENCE_TOKEN_ENV,
    derive_camera_token,
)
from vehicle_intelligence.infrastructure.persistence.jsonl import JsonlVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.vision.remote import (
    RemotePlateDetector,
    RemoteVehicleDetector,
)
from vehicle_intelligence.interfaces import composition
from vehicle_intelligence.interfaces.api import create_app
from vehicle_intelligence.interfaces.composition import _detectors, _repository


class _InitializationFailureRepository(InMemoryVehicleEventRepository):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def ensure_indexes(self) -> None:
        raise RuntimeError("simulated initialization failure")

    async def close(self) -> None:
        self.closed = True


class _InitializationAndCleanupFailureRepository(_InitializationFailureRepository):
    async def close(self) -> None:
        self.closed = True
        raise RuntimeError("simulated cleanup failure")


async def test_event_repository_uses_only_the_configured_persistent_backend(tmp_path) -> None:
    base = load_settings()
    local_settings = base.model_copy(
        update={
            "storage": base.storage.model_copy(update={"output_directory": tmp_path}),
        }
    )
    local = _repository(local_settings)
    assert isinstance(local, JsonlVehicleEventRepository)

    mongo_settings = base.model_copy(
        update={"mongodb": base.mongodb.model_copy(update={"enabled": True})}
    )
    mongo = _repository(mongo_settings)
    try:
        assert isinstance(mongo, MongoVehicleEventRepository)
    finally:
        await mongo.close()


def test_api_closes_resources_when_early_initialization_fails() -> None:
    repository = _InitializationFailureRepository()
    app = create_app(load_settings(), repository)

    with pytest.raises(RuntimeError, match="simulated initialization failure"), TestClient(app):
        pass

    assert repository.closed is True


def test_api_cleanup_failure_does_not_mask_initialization_failure() -> None:
    repository = _InitializationAndCleanupFailureRepository()
    app = create_app(load_settings(), repository)

    with pytest.raises(RuntimeError, match="simulated initialization failure"), TestClient(app):
        pass

    assert repository.closed is True


def test_detector_composition_uses_remote_adapters_only_when_scheduler_enabled(
    monkeypatch,
) -> None:
    base = load_settings()
    scheduler = base.gpu_scheduler.model_copy(update={"enabled": True})
    settings = base.model_copy(update={"gpu_scheduler": scheduler})
    master = "m" * 32
    monkeypatch.setenv(INFERENCE_SOCKET_ENV, str(scheduler.socket_path))
    monkeypatch.setenv(INFERENCE_CAMERA_ENV, settings.camera.id)
    monkeypatch.setenv(
        INFERENCE_TOKEN_ENV,
        derive_camera_token(master, settings.camera.id),
    )

    vehicle, plate = _detectors(settings)

    assert isinstance(vehicle, RemoteVehicleDetector)
    assert isinstance(plate, RemotePlateDetector)
    assert INFERENCE_TOKEN_ENV not in __import__("os").environ


async def test_pipeline_starts_outbox_when_publisher_is_initially_unavailable(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Publisher:
        async def initialize(self) -> None:
            calls.append("publisher.initialize")
            raise EventBusError("injected outage")

        async def publish(self, event) -> bool:
            del event
            return True

        async def close(self) -> None:
            calls.append("publisher.close")

    class Outbox:
        async def initialize(self) -> None:
            calls.append("outbox.initialize")

        async def stage(self, event, media) -> None:
            del event, media

        async def close(self) -> None:
            calls.append("outbox.close")

    class Source:
        source_id = "test"
        source_fps = 1.0

        def frames(self):
            return iter(())

        def close(self) -> None:
            calls.append("source.close")

    class Pipeline:
        async def run(self) -> PipelineResult:
            calls.append("pipeline.run")
            return PipelineResult((), PipelineStats())

    publisher = Publisher()
    outbox = Outbox()
    monkeypatch.setattr(composition, "_publisher", lambda settings: publisher)
    monkeypatch.setattr(composition, "_media_storage", lambda settings: object())
    monkeypatch.setattr(
        composition,
        "_finalization_outbox",
        lambda settings, media_storage, event_publisher: outbox,
    )
    monkeypatch.setattr(composition, "_live_preview", lambda settings: None)
    monkeypatch.setattr(composition, "_pipeline", lambda *args, **kwargs: Pipeline())

    await composition.execute_pipeline(load_settings(), Source())

    assert calls == [
        "publisher.initialize",
        "outbox.initialize",
        "pipeline.run",
        "source.close",
        "outbox.close",
        "publisher.close",
    ]


async def test_pipeline_cleanup_preserves_cancellation_and_attempts_every_closer(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Publisher:
        async def initialize(self) -> None:
            calls.append("publisher.initialize")

        async def publish(self, event) -> bool:
            del event
            return True

        async def close(self) -> None:
            calls.append("publisher.close")

    class Outbox:
        async def initialize(self) -> None:
            calls.append("outbox.initialize")

        async def stage(self, event, media) -> None:
            del event, media

        async def close(self) -> None:
            calls.append("outbox.close")
            raise RuntimeError("secondary outbox cleanup failure")

    class LivePreview:
        async def initialize(self) -> None:
            calls.append("live.initialize")

        async def close(self) -> None:
            calls.append("live.close")

    class Source:
        source_id = "test"
        source_fps = 1.0

        def frames(self):
            return iter(())

        def close(self) -> None:
            calls.append("source.close")
            raise RuntimeError("secondary source cleanup failure")

    class Pipeline:
        async def run(self) -> PipelineResult:
            calls.append("pipeline.run")
            raise asyncio.CancelledError("primary pipeline cancellation")

    publisher = Publisher()
    outbox = Outbox()
    live_preview = LivePreview()
    monkeypatch.setattr(composition, "_publisher", lambda settings: publisher)
    monkeypatch.setattr(composition, "_media_storage", lambda settings: object())
    monkeypatch.setattr(
        composition,
        "_finalization_outbox",
        lambda settings, media_storage, event_publisher: outbox,
    )
    monkeypatch.setattr(composition, "_live_preview", lambda settings: live_preview)
    monkeypatch.setattr(composition, "_pipeline", lambda *args, **kwargs: Pipeline())

    with pytest.raises(asyncio.CancelledError, match="primary pipeline cancellation"):
        await composition.execute_pipeline(load_settings(), Source())

    assert calls == [
        "publisher.initialize",
        "outbox.initialize",
        "live.initialize",
        "pipeline.run",
        "source.close",
        "live.close",
        "outbox.close",
        "publisher.close",
    ]


async def test_pipeline_constructor_failure_closes_every_resource_already_owned(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Publisher:
        async def initialize(self) -> None:
            raise AssertionError("publisher initialization must not run")

        async def publish(self, event) -> bool:
            del event
            return True

        async def close(self) -> None:
            calls.append("publisher.close")

    class Media:
        async def put(self, key: str, data: bytes, content_type: str) -> str:
            del data, content_type
            return key

        async def close(self) -> None:
            calls.append("media.close")

    class Source:
        source_id = "test"
        source_fps = 1.0

        def frames(self):
            return iter(())

        def close(self) -> None:
            calls.append("source.close")

    publisher = Publisher()
    media = Media()
    monkeypatch.setattr(composition, "_publisher", lambda settings: publisher)
    monkeypatch.setattr(composition, "_media_storage", lambda settings: media)

    def fail_outbox(*args):
        del args
        raise RuntimeError("simulated outbox constructor failure")

    monkeypatch.setattr(composition, "_finalization_outbox", fail_outbox)

    with pytest.raises(RuntimeError, match="simulated outbox constructor failure"):
        await composition.execute_pipeline(load_settings(), Source())

    assert calls == ["source.close", "publisher.close", "media.close"]


async def test_pipeline_invalid_delivery_budget_still_closes_owned_source(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class Source:
        source_id = "test"
        source_fps = 1.0

        def frames(self):
            return iter(())

        def close(self) -> None:
            calls.append("source.close")

    settings = load_settings()
    invalid = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"backend": "minio"}),
            "event_bus": settings.event_bus.model_copy(update={"backend": "redis"}),
            "redis": settings.redis.model_copy(update={"block_ms": 60_000}),
        }
    )
    monkeypatch.setattr(
        composition,
        "_publisher",
        lambda settings: pytest.fail("publisher must not be constructed"),
    )

    with pytest.raises(ConfigurationError, match="first-delivery HTTP and publisher budget"):
        await composition.execute_pipeline(invalid, Source())

    assert calls == ["source.close"]
