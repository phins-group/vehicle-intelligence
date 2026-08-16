"""Role-based composition for pinned sample source adapters."""

from __future__ import annotations

from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.http import BootstrapHttpClient
from vehicle_intelligence.training.bootstrap.huggingface import (
    SOURCE_INFO as PLATE_SOURCE_INFO,
)
from vehicle_intelligence.training.bootstrap.huggingface import (
    HuggingFacePlateSampleSource,
)
from vehicle_intelligence.training.bootstrap.open_images import (
    SOURCE_INFO as VEHICLE_SOURCE_INFO,
)
from vehicle_intelligence.training.bootstrap.open_images import (
    OpenImagesVehicleSampleSource,
)
from vehicle_intelligence.training.domain import DetectorRole


def acquire_bootstrap_samples(
    role: DetectorRole,
    *,
    samples_per_class: int,
    http: BootstrapHttpClient,
) -> tuple[BootstrapSourceInfo, list[AcquiredDetectorSample]]:
    if role is DetectorRole.VEHICLE:
        return (
            VEHICLE_SOURCE_INFO,
            OpenImagesVehicleSampleSource(http).acquire(samples_per_class=samples_per_class),
        )
    return (
        PLATE_SOURCE_INFO,
        HuggingFacePlateSampleSource(http).acquire(sample_count=samples_per_class),
    )
