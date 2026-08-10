#!/usr/bin/env python3
"""Evaluate an immutable OCR feedback export and optionally enforce gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_intelligence.application.dataset_evaluation import (
    evaluate_ocr_records,
    evaluation_to_jsonable,
    release_gates,
)
from vehicle_intelligence.application.dataset_export import verify_dataset_export
from vehicle_intelligence.exceptions import VehicleIntelligenceError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an exported OCR feedback dataset")
    parser.add_argument("export_directory", type=Path)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--minimum-exact-accuracy", type=float)
    parser.add_argument("--minimum-character-accuracy", type=float)
    parser.add_argument("--maximum-ece", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def run(args: argparse.Namespace) -> int:
    manifest, manifest_sha256 = verify_dataset_export(args.export_directory)
    labels = args.export_directory.expanduser().resolve() / "labels.jsonl"
    records = [json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    evaluation = evaluate_ocr_records(records, args.bins)
    payload = evaluation_to_jsonable(evaluation)
    payload["exportId"] = manifest["exportId"]
    payload["manifestSha256"] = manifest_sha256
    failures = release_gates(
        evaluation,
        minimum_exact_accuracy=args.minimum_exact_accuracy,
        minimum_character_accuracy=args.minimum_character_accuracy,
        maximum_ece=args.maximum_ece,
    )
    payload["releaseGate"] = {"passed": not failures, "failures": failures}
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return 0 if not failures else 4


def main() -> None:
    parser = _parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (VehicleIntelligenceError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
