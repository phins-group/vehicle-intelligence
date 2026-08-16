"""Pure benchmark statistics and regression gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def latency_summary(values_ms: list[float]) -> dict[str, float | int]:
    if not values_ms:
        return {"count": 0}
    samples = np.asarray(values_ms, dtype=np.float64)
    return {
        "count": len(values_ms),
        "meanMs": round(float(samples.mean()), 3),
        "p50Ms": round(float(np.percentile(samples, 50)), 3),
        "p95Ms": round(float(np.percentile(samples, 95)), 3),
        "p99Ms": round(float(np.percentile(samples, 99)), 3),
        "maxMs": round(float(samples.max()), 3),
    }


@dataclass(frozen=True, slots=True)
class BenchmarkGate:
    maximum_p95_regression_percent: float = 15.0
    minimum_throughput_ratio: float = 0.90

    def __post_init__(self) -> None:
        if self.maximum_p95_regression_percent < 0:
            raise ValueError("maximum p95 regression cannot be negative")
        if not 0 < self.minimum_throughput_ratio <= 1:
            raise ValueError("minimum throughput ratio must be in (0, 1]")


def compare_detector_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    gate: BenchmarkGate,
) -> tuple[str, ...]:
    """Return stable failure reasons instead of hiding regressions in logs."""

    baseline_p95 = _positive_number(baseline, "latency", "p95Ms")
    candidate_p95 = _positive_number(candidate, "latency", "p95Ms")
    baseline_fps = _positive_number(baseline, "effectiveFps")
    candidate_fps = _positive_number(candidate, "effectiveFps")
    failures: list[str] = []
    p95_regression = ((candidate_p95 / baseline_p95) - 1) * 100
    if p95_regression > gate.maximum_p95_regression_percent:
        failures.append(
            f"p95 latency regressed {p95_regression:.2f}% "
            f"(limit {gate.maximum_p95_regression_percent:.2f}%)"
        )
    throughput_ratio = candidate_fps / baseline_fps
    if throughput_ratio < gate.minimum_throughput_ratio:
        failures.append(
            f"throughput ratio {throughput_ratio:.3f} is below {gate.minimum_throughput_ratio:.3f}"
        )
    return tuple(failures)


def _positive_number(document: dict[str, Any], *path: str) -> float:
    value: object = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"benchmark report is missing {'.'.join(path)}")
        value = value[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"benchmark field {'.'.join(path)} must be positive")
    return float(value)
