#!/usr/bin/env python3
"""Verify persisted detector and edge-capacity benchmark gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify all persisted performance gates")
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("output/benchmarks"),
    )
    parser.add_argument("--output", type=Path)
    return parser


def verify(directory: Path) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    failures: list[str] = []
    kinds: set[str] = set()
    accuracy_representative = False
    for path in sorted(directory.glob("*.json")):
        try:
            document = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{path.name}:invalid_json:{type(exc).__name__}")
            continue
        gate = document.get("gate")
        if not isinstance(gate, dict):
            continue
        kind = str(document.get("kind", "unknown"))
        kinds.add(kind)
        model = document.get("model") or {}
        result = document.get("result") or {}
        latency = document.get("latency") or result.get("endToEndLatency") or {}
        passed = gate.get("passed") is True
        sha256 = model.get("sha256")
        if not isinstance(sha256, str) or len(sha256) != 64:
            failures.append(f"{path.name}:invalid_model_hash")
            passed = False
        if int(latency.get("count", 0)) <= 0:
            failures.append(f"{path.name}:missing_latency_samples")
            passed = False
        if gate.get("passed") is not True:
            failures.append(f"{path.name}:gate_failed")
        input_document = document.get("input") or {}
        accuracy_representative = accuracy_representative or (
            kind == "detector-runtime-benchmark"
            and input_document.get("accuracyRepresentative") is True
            and passed
        )
        accepted.append(
            {
                "file": path.name,
                "kind": kind,
                "provider": document.get("provider") or model.get("provider"),
                "passed": passed,
                "p95Ms": latency.get("p95Ms"),
            }
        )
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


def main() -> None:
    args = _parser().parse_args()
    report = verify(args.directory)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(0 if report["passed"] else 4)


if __name__ == "__main__":
    main()
