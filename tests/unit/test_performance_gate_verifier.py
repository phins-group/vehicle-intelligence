import json
import math

import pytest

from scripts.verify_performance_gates import verify


def _result(kind: str, *, representative: bool = False, passed: bool = True):
    document = {
        "kind": kind,
        "model": {"sha256": "a" * 64, "provider": "test"},
        "gate": {"passed": passed, "failures": []},
    }
    if kind == "detector-runtime-benchmark":
        document.update(
            {
                "input": {"accuracyRepresentative": representative},
                "latency": {"count": 5, "p95Ms": 10},
            }
        )
    else:
        document.update(
            {
                "scheduler": {"dropRatio": 0.01},
                "result": {"endToEndLatency": {"count": 5, "p95Ms": 20}},
            }
        )
    return document


def test_performance_verifier_requires_both_gate_kinds_and_real_input(tmp_path) -> None:
    (tmp_path / "detector.json").write_text(
        json.dumps(_result("detector-runtime-benchmark", representative=True))
    )
    (tmp_path / "edge.json").write_text(json.dumps(_result("edge-capacity-benchmark")))

    assert verify(tmp_path)["passed"]

    (tmp_path / "edge.json").write_text(
        json.dumps(_result("edge-capacity-benchmark", passed=False))
    )
    failed = verify(tmp_path)
    assert not failed["passed"]
    assert "edge.json:gate_failed" in failed["failures"]


def test_performance_verifier_enforces_global_edge_limits(tmp_path) -> None:
    (tmp_path / "detector.json").write_text(
        json.dumps(_result("detector-runtime-benchmark", representative=True))
    )
    overloaded = _result("edge-capacity-benchmark")
    overloaded["scheduler"]["dropRatio"] = 0.55
    overloaded["result"]["endToEndLatency"]["p95Ms"] = 590
    (tmp_path / "edge.json").write_text(json.dumps(overloaded))

    report = verify(tmp_path)

    assert not report["passed"]
    assert any("drop_ratio_exceeds_limit" in failure for failure in report["failures"])
    assert any("p95_latency_exceeds_limit" in failure for failure in report["failures"])


def test_performance_verifier_rejects_non_finite_numbers_and_limits(tmp_path) -> None:
    (tmp_path / "detector.json").write_text(
        json.dumps(_result("detector-runtime-benchmark", representative=True))
    )
    edge = _result("edge-capacity-benchmark")
    edge["scheduler"]["dropRatio"] = math.nan
    (tmp_path / "edge.json").write_text(json.dumps(edge))

    report = verify(tmp_path)

    assert not report["passed"]
    assert any("invalid_json" in failure for failure in report["failures"])
    with pytest.raises(ValueError, match="limits are invalid"):
        verify(tmp_path, maximum_edge_p95_ms=math.inf)


def test_performance_verifier_rejects_out_of_range_metrics(tmp_path) -> None:
    (tmp_path / "detector.json").write_text(
        json.dumps(_result("detector-runtime-benchmark", representative=True))
    )
    edge = _result("edge-capacity-benchmark")
    edge["scheduler"]["dropRatio"] = -0.01
    edge["result"]["endToEndLatency"]["p95Ms"] = -1
    (tmp_path / "edge.json").write_text(json.dumps(edge))

    report = verify(tmp_path)

    assert not report["passed"]
    assert "edge.json:invalid_drop_ratio" in report["failures"]
    assert "edge.json:invalid_p95_latency" in report["failures"]
