from __future__ import annotations

import pytest

from vehicle_intelligence.config import (
    AppConfig,
    DetectorConfig,
    VehicleDetectorConfig,
    load_settings,
)
from vehicle_intelligence.exceptions import ConfigurationError, UnsupportedDetectorProvider
from vehicle_intelligence.infrastructure.vision import factory
from vehicle_intelligence.interfaces.composition import validate_runtime_settings


def vehicle_config(provider: str) -> VehicleDetectorConfig:
    return VehicleDetectorConfig(
        provider=provider,
        model_path="vehicle.onnx",
        model_name="vehicle",
        model_version="1",
        confidence=0.4,
        iou=0.5,
        model_classes=["car"],
        classes=["car"],
    )


def plate_config(provider: str) -> DetectorConfig:
    return DetectorConfig(
        provider=provider,
        model_path="plate.onnx",
        model_name="plate",
        model_version="1",
        confidence=0.3,
        iou=0.45,
        model_classes=["license_plate"],
    )


def test_factory_routes_yolo_and_picodet_without_loading_models(monkeypatch) -> None:
    yolo = object()
    picodet = object()
    monkeypatch.setattr(factory, "UltralyticsVehicleDetector", lambda _config: yolo)
    monkeypatch.setattr(factory, "PicoDetDetector", lambda _config: picodet)

    assert factory.create_vehicle_detector(vehicle_config("yolo")) is yolo
    assert factory.create_vehicle_detector(vehicle_config("ultralytics")) is yolo
    assert factory.create_vehicle_detector(vehicle_config("picodet")) is picodet


@pytest.mark.parametrize(
    ("vehicle_provider", "plate_provider"),
    (("picodet", "yolo"), ("yolo", "picodet")),
)
def test_vehicle_and_plate_providers_are_composed_independently(
    monkeypatch,
    vehicle_provider: str,
    plate_provider: str,
) -> None:
    monkeypatch.setattr(
        factory,
        "UltralyticsVehicleDetector",
        lambda _config: "yolo-vehicle",
    )
    monkeypatch.setattr(
        factory,
        "UltralyticsPlateDetector",
        lambda _config: "yolo-plate",
    )
    monkeypatch.setattr(factory, "PicoDetDetector", lambda _config: "picodet-vehicle")
    monkeypatch.setattr(
        factory,
        "PicoDetPlateDetector",
        lambda _config: "picodet-plate",
    )

    vehicle = factory.create_vehicle_detector(vehicle_config(vehicle_provider))
    plate = factory.create_plate_detector(plate_config(plate_provider))

    assert vehicle == f"{vehicle_provider}-vehicle"
    assert plate == f"{plate_provider}-plate"


def test_factory_rejects_unknown_provider_with_common_exception() -> None:
    with pytest.raises(UnsupportedDetectorProvider, match="vehicle detector"):
        factory.create_vehicle_detector(vehicle_config("abc"))
    with pytest.raises(UnsupportedDetectorProvider, match="plate detector"):
        factory.create_plate_detector(plate_config("abc"))


@pytest.mark.parametrize(
    ("vehicle_provider", "plate_provider"),
    (("picodet", "yolo"), ("yolo", "picodet")),
)
def test_runtime_configuration_accepts_both_hybrid_combinations(
    vehicle_provider: str,
    plate_provider: str,
) -> None:
    settings = load_settings()
    vision = settings.vision.model_copy(
        update={
            "vehicle_detection": settings.vision.vehicle_detection.model_copy(
                update={"provider": vehicle_provider}
            ),
            "plate_detection": settings.vision.plate_detection.model_copy(
                update={"provider": plate_provider, "model_path": "plate-model"}
            ),
        }
    )

    validate_runtime_settings(settings.model_copy(update={"vision": vision}))


def test_plate_only_runtime_does_not_validate_unused_vehicle_provider() -> None:
    settings = load_settings()
    vision = settings.vision.model_copy(
        update={
            "plate_only": True,
            "vehicle_detection": settings.vision.vehicle_detection.model_copy(
                update={"provider": "not-installed"}
            ),
            "plate_detection": settings.vision.plate_detection.model_copy(
                update={"provider": "yolo", "model_path": "plate-model"}
            ),
        }
    )

    validate_runtime_settings(settings.model_copy(update={"vision": vision}))


@pytest.mark.parametrize("environment", ("production", "Production", " production "))
def test_production_runtime_refuses_paddle_managed_model_downloads(environment: str) -> None:
    settings = load_settings()
    vision = settings.vision.model_copy(
        update={
            "plate_detection": settings.vision.plate_detection.model_copy(
                update={"provider": "yolo", "model_path": "plate-model"}
            )
        }
    )
    app = AppConfig().model_copy(update={"environment": environment})
    production = settings.model_copy(update={"app": app, "vision": vision})

    with pytest.raises(ConfigurationError, match="production OCR requires local"):
        validate_runtime_settings(production)


def test_camera_runtime_rejects_minio_delivery_budget_above_outbox_deadline() -> None:
    settings = load_settings()
    invalid = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"backend": "minio"}),
            "event_bus": settings.event_bus.model_copy(update={"backend": "redis"}),
            "redis": settings.redis.model_copy(update={"block_ms": 60_000}),
        }
    )

    with pytest.raises(ConfigurationError, match="first-delivery HTTP and publisher budget"):
        validate_runtime_settings(invalid)
