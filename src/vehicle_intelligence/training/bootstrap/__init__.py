"""License-traced sample acquisition for detector pipeline bootstrap only."""

from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapBuildResult,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.bootstrap.service import acquire_bootstrap_samples
from vehicle_intelligence.training.bootstrap.writer import (
    BootstrapSourceWriter,
    verify_bootstrap_source,
)

__all__ = [
    "AcquiredDetectorSample",
    "BootstrapBuildResult",
    "BootstrapSourceInfo",
    "BootstrapSourceWriter",
    "acquire_bootstrap_samples",
    "verify_bootstrap_source",
]
