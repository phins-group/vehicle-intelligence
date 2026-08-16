#!/usr/bin/env python3
"""Verify persisted detector and edge-capacity benchmark gate evidence."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class _GateEvidence:
    kind: str
    accepted: dict[str, Any]
    failures: tuple[str, ...]
    accuracy_representative: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify all persisted performance gates")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("output/benchmarks"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--maximum-edge-drop-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-edge-p95-ms", type=float, default=250.0)
    return parser


def verify(
    directory: Path,
    *,
    maximum_edge_drop_ratio: float = 0.10,
    maximum_edge_p95_ms: float = 250.0,
) -> dict[str, Any]:
    _validate_limits(maximum_edge_drop_ratio, maximum_edge_p95_ms)
    accepted: list[dict[str, Any]] = []
    failures: list[str] = []
    kinds: set[str] = set()
    accuracy_representative = False
    for path in sorted(directory.glob("*.json")):
        document = _load_document(path, failures)
        if document is None or not isinstance(document.get("gate"), dict):
            continue
        evidence = _evaluate_gate(
            path,
            document,
            maximum_edge_drop_ratio=maximum_edge_drop_ratio,
            maximum_edge_p95_ms=maximum_edge_p95_ms,
        )
        kinds.add(evidence.kind)
        failures.extend(evidence.failures)
        accepted.append(evidence.accepted)
        accuracy_representative = accuracy_representative or evidence.accuracy_representative
    required = {"detector-runtime-benchmark", "edge-capacity-benchmark"}
    missing = sorted(required - kinds)
    failures.extend(f"missing_kind:{kind}" for kind in missing)
    if not accuracy_representative:
        failures.append("missing_accuracy_representative_detector_gate")
    return {
        "schemaVersion": 1,
        "directory": str(directory.resolve()),
        "accepted": accepted,
        "passed": not failures,
        "failures": failures,
    }


def _validate_limits(maximum_edge_drop_ratio: float, maximum_edge_p95_ms: float) -> None:
    if not (
        math.isfinite(maximum_edge_drop_ratio)
        and 0 <= maximum_edge_drop_ratio <= 1
        and math.isfinite(maximum_edge_p95_ms)
        and maximum_edge_p95_ms > 0
    ):
        raise ValueError("performance verification limits are invalid")


def _load_document(path: Path, failures: list[str]) -> dict[str, Any] | None:
    try:
        document = json.loads(
            path.read_text(),
            parse_constant=_reject_non_json_number,
        )
    except (OSError, ValueError) as exc:
        failures.append(f"{path.name}:invalid_json:{type(exc).__name__}")
        return None
    if not isinstance(document, dict):
        failures.append(f"{path.name}:invalid_document")
        return None
    return document


def _evaluate_gate(
    path: Path,
    document: dict[str, Any],
    *,
    maximum_edge_drop_ratio: float,
    maximum_edge_p95_ms: float,
) -> _GateEvidence:
    gate = _mapping(document.get("gate"))
    model = _mapping(document.get("model"))
    result = _mapping(document.get("result"))
    latency = _mapping(document.get("latency") or result.get("endToEndLatency"))
    kind = str(document.get("kind", "unknown"))
    passed = gate.get("passed") is True
    failures: list[str] = []

    if not _valid_sha256(model.get("sha256")):
        failures.append(f"{path.name}:invalid_model_hash")
        passed = False
    latency_count = _number(latency.get("count"))
    if latency_count is None or latency_count <= 0 or not latency_count.is_integer():
        failures.append(f"{path.name}:missing_latency_samples")
        passed = False
    p95_ms = _number(latency.get("p95Ms"))
    if p95_ms is None or p95_ms < 0:
        failures.append(f"{path.name}:invalid_p95_latency")
        passed = False

    drop_ratio: float | None = None
    if kind == "edge-capacity-benchmark":
        passed, drop_ratio = _evaluate_edge_metrics(
            path,
            document,
            failures,
            passed=passed,
            p95_ms=p95_ms,
            maximum_drop_ratio=maximum_edge_drop_ratio,
            maximum_p95_ms=maximum_edge_p95_ms,
        )
    if gate.get("passed") is not True:
        failures.append(f"{path.name}:gate_failed")

    input_document = _mapping(document.get("input"))
    representative = (
        kind == "detector-runtime-benchmark"
        and input_document.get("accuracyRepresentative") is True
        and passed
    )
    return _GateEvidence(
        kind=kind,
        accepted={
            "file": path.name,
            "kind": kind,
            "provider": document.get("provider") or model.get("provider"),
            "passed": passed,
            "p95Ms": p95_ms,
            "dropRatio": drop_ratio,
        },
        failures=tuple(failures),
        accuracy_representative=representative,
    )


def _evaluate_edge_metrics(
    path: Path,
    document: dict[str, Any],
    failures: list[str],
    *,
    passed: bool,
    p95_ms: float | None,
    maximum_drop_ratio: float,
    maximum_p95_ms: float,
) -> tuple[bool, float | None]:
    scheduler = _mapping(document.get("scheduler"))
    drop_ratio = _number(scheduler.get("dropRatio"))
    if drop_ratio is None:
        failures.append(f"{path.name}:missing_drop_ratio")
        passed = False
    elif not 0 <= drop_ratio <= 1:
        failures.append(f"{path.name}:invalid_drop_ratio")
        passed = False
    elif drop_ratio > maximum_drop_ratio:
        failures.append(
            f"{path.name}:drop_ratio_exceeds_limit:{drop_ratio:.6f}>{maximum_drop_ratio:.6f}"
        )
        passed = False
    if p95_ms is not None and p95_ms >= 0 and p95_ms > maximum_p95_ms:
        failures.append(f"{path.name}:p95_latency_exceeds_limit:{p95_ms:.3f}>{maximum_p95_ms:.3f}")
        passed = False
    return passed, drop_ratio


def _mapping(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _reject_non_json_number(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def main() -> None:
    args = _parser().parse_args()
    report = verify(
        args.directory,
        maximum_edge_drop_ratio=args.maximum_edge_drop_ratio,
        maximum_edge_p95_ms=args.maximum_edge_p95_ms,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 4)


if __name__ == "__main__":
    main()
