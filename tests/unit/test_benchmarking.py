import pytest

from vehicle_intelligence.application.benchmarking import (
    BenchmarkGate,
    compare_detector_reports,
    latency_summary,
)


def test_latency_summary_reports_tail_percentiles() -> None:
    summary = latency_summary([1, 2, 3, 4, 10])
    assert summary["count"] == 5
    assert summary["p50Ms"] == 3
    assert summary["p95Ms"] == pytest.approx(8.8)
    assert summary["p99Ms"] == pytest.approx(9.76)


def test_benchmark_gate_reports_latency_and_throughput_regressions() -> None:
    baseline = {"latency": {"p95Ms": 10.0}, "effectiveFps": 100.0}
    candidate = {"latency": {"p95Ms": 12.0}, "effectiveFps": 80.0}
    failures = compare_detector_reports(
        baseline,
        candidate,
        BenchmarkGate(maximum_p95_regression_percent=10, minimum_throughput_ratio=0.9),
    )
    assert len(failures) == 2
    assert "p95 latency" in failures[0]
    assert "throughput ratio" in failures[1]


def test_benchmark_gate_accepts_candidate_within_bounds() -> None:
    baseline = {"latency": {"p95Ms": 10.0}, "effectiveFps": 100.0}
    candidate = {"latency": {"p95Ms": 10.5}, "effectiveFps": 95.0}
    assert not compare_detector_reports(baseline, candidate, BenchmarkGate())
