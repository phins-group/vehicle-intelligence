import json

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
        document["result"] = {"endToEndLatency": {"count": 5, "p95Ms": 20}}
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
