"""Config-driven detector composition without leaking provider SDKs upstream."""

from __future__ import annotations

from vehicle_intelligence.application.ports import PlateDetector, VehicleDetector
from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.exceptions import UnsupportedDetectorProvider
from vehicle_intelligence.infrastructure.vision.onnx_runtime import (
    OnnxRuntimePlateDetector,
    OnnxRuntimeVehicleDetector,
)
from vehicle_intelligence.infrastructure.vision.picodet import (
    PicoDetDetector,
    PicoDetPlateDetector,
)
from vehicle_intelligence.infrastructure.vision.ultralytics import (
    UltralyticsPlateDetector,
    UltralyticsVehicleDetector,
)

_YOLO_PROVIDERS = frozenset({"ultralytics", "yolo"})
_ONNX_PROVIDERS = frozenset({"onnx", "onnxruntime", "tensorrt"})
SUPPORTED_DETECTOR_PROVIDERS = _YOLO_PROVIDERS | _ONNX_PROVIDERS | {"picodet"}


def validate_detector_provider(provider: str, component: str = "detector") -> str:
    """Normalize and validate a configured provider without loading its model."""

    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_DETECTOR_PROVIDERS:
        raise UnsupportedDetectorProvider(f"unsupported {component} provider: {provider}")
    return normalized


def create_vehicle_detector(config: VehicleDetectorConfig) -> VehicleDetector:
    provider = validate_detector_provider(config.provider, "vehicle detector")
    if provider in _YOLO_PROVIDERS:
        return UltralyticsVehicleDetector(config)
    if provider == "picodet":
        return PicoDetDetector(config)
    if provider in _ONNX_PROVIDERS:
        effective = _with_tensorrt(config) if provider == "tensorrt" else config
        return OnnxRuntimeVehicleDetector(effective)
    raise AssertionError("validated vehicle detector provider was not composed")


def create_plate_detector(config: DetectorConfig) -> PlateDetector:
    provider = validate_detector_provider(config.provider, "plate detector")
    if provider in _YOLO_PROVIDERS:
        return UltralyticsPlateDetector(config)
    if provider == "picodet":
        return PicoDetPlateDetector(config)
    if provider in _ONNX_PROVIDERS:
        effective = _with_tensorrt(config) if provider == "tensorrt" else config
        return OnnxRuntimePlateDetector(effective)
    raise AssertionError("validated plate detector provider was not composed")


def _with_tensorrt(config: DetectorConfig) -> DetectorConfig:
    if config.execution_providers:
        return config
    return config.model_copy(update={"execution_providers": ["tensorrt"]})
