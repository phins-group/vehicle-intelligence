#!/usr/bin/env python3
"""Benchmark one real detector artifact/provider with machine-readable gates."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.application.benchmarking import (
    BenchmarkGate,
    compare_detector_reports,
    latency_summary,
)
from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig
from vehicle_intelligence.infrastructure.vision.factory import (
    create_plate_detector,
    create_vehicle_detector,
)
from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a detector execution provider")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("yolo", "ultralytics", "picodet", "onnxruntime", "tensorrt"),
        required=True,
    )
    parser.add_argument("--role", choices=("vehicle", "plate"), required=True)
    parser.add_argument("--model-name", default="benchmark-detector")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--model-hash")
    parser.add_argument("--image", type=Path)
    parser.add_argument(
        "--representative-input",
        action="store_true",
        help="assert that the supplied image belongs to the deployment benchmark set",
    )
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--synthetic-width", type=int, default=1920)
    parser.add_argument("--synthetic-height", type=int, default=1080)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--device")
    parser.add_argument("--execution-provider", action="append", default=[])
    parser.add_argument("--output-format", choices=("auto", "raw", "nms"), default="auto")
    parser.add_argument("--model-classes", help="comma-separated model output classes")
    parser.add_argument(
        "--allowed-classes", default="car,motorcycle,bus,truck", help="vehicle classes to retain"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--maximum-p95-regression-percent", type=float, default=15.0)
    parser.add_argument("--minimum-throughput-ratio", type=float, default=0.90)
    parser.add_argument("--minimum-fps", type=float)
    parser.add_argument("--maximum-p95-ms", type=float)
    parser.add_argument("--allow-cross-machine-baseline", action="store_true")
    return parser


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    result = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not result:
        raise SystemExit("class lists cannot be empty")
    return result


def _input_image(args: argparse.Namespace) -> tuple[np.ndarray, dict[str, object]]:
    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"cannot decode benchmark image: {args.image}")
        return image, {
            "kind": "image",
            "path": str(args.image.resolve()),
            "width": image.shape[1],
            "height": image.shape[0],
            "accuracyRepresentative": args.representative_input,
        }
    if args.synthetic_width < 1 or args.synthetic_height < 1:
        raise SystemExit("synthetic dimensions must be positive")
    y, x = np.indices((args.synthetic_height, args.synthetic_width), dtype=np.uint16)
    image = np.stack(((x % 256), (y % 256), ((x + y) % 256)), axis=2).astype(np.uint8)
    return image, {
        "kind": "deterministic-synthetic",
        "width": image.shape[1],
        "height": image.shape[0],
        "accuracyRepresentative": False,
    }


def _peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return round(value / divisor, 3)


def _machine() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup < 0 or args.iterations < 1:
        raise SystemExit("warmup must be non-negative and iterations must be positive")
    if args.image_size < 1 or not 0 <= args.confidence <= 1 or not 0 <= args.iou <= 1:
        raise SystemExit("invalid detector benchmark configuration")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"model artifact does not exist: {model_path}")
    common: dict[str, object] = {
        "provider": args.provider,
        "model_path": str(model_path),
        "model_name": args.model_name,
        "model_version": args.model_version,
        "model_hash": args.model_hash,
        "confidence": args.confidence,
        "iou": args.iou,
        "image_size": args.image_size,
        "device": args.device,
        "execution_providers": args.execution_provider,
        "onnx_output_format": args.output_format,
        "model_classes": _csv(args.model_classes),
    }
    load_started = time.perf_counter()
    if args.role == "vehicle":
        config = VehicleDetectorConfig(**common, classes=_csv(args.allowed_classes))
        detector = create_vehicle_detector(config)
    else:
        config = DetectorConfig(**common)
        detector = create_plate_detector(config)
    load_ms = (time.perf_counter() - load_started) * 1000
    image, input_metadata = _input_image(args)
    for _ in range(args.warmup):
        detector.detect(image)
    samples: list[float] = []
    detection_counts: list[int] = []
    wall_started = time.perf_counter()
    for _ in range(args.iterations):
        started = time.perf_counter()
        detections = detector.detect(image)
        samples.append((time.perf_counter() - started) * 1000)
        detection_counts.append(len(detections))
    wall_seconds = time.perf_counter() - wall_started
    effective_fps = args.iterations / max(wall_seconds, 1e-9)
    runtime_providers = list(getattr(detector, "execution_providers", (args.device or "auto",)))
    report: dict[str, Any] = {
        "schemaVersion": 1,
        "kind": "detector-runtime-benchmark",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "machine": _machine(),
        "role": args.role,
        "provider": args.provider,
        "executionProviders": runtime_providers,
        "model": {
            "name": args.model_name,
            "version": args.model_version,
            "sha256": sha256_file(model_path),
            "sizeBytes": model_path.stat().st_size,
        },
        "input": input_metadata,
        "warmupIterations": args.warmup,
        "measuredIterations": args.iterations,
        "loadMs": round(load_ms, 3),
        "effectiveFps": round(effective_fps, 3),
        "latency": latency_summary(samples),
        "detections": {
            "minimum": min(detection_counts),
            "maximum": max(detection_counts),
            "mean": round(sum(detection_counts) / len(detection_counts), 3),
        },
        "peakRssMb": _peak_rss_mb(),
        "gate": {"passed": True, "failures": []},
    }
    failures: list[str] = []
    if args.minimum_fps is not None and effective_fps < args.minimum_fps:
        failures.append(f"effective FPS {effective_fps:.3f} is below {args.minimum_fps:.3f}")
    p95_ms = float(report["latency"]["p95Ms"])
    if args.maximum_p95_ms is not None and p95_ms > args.maximum_p95_ms:
        failures.append(f"p95 latency {p95_ms:.3f}ms exceeds {args.maximum_p95_ms:.3f}ms")
    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if not args.allow_cross_machine_baseline and baseline.get("machine") != report["machine"]:
            raise SystemExit("baseline machine does not match; use --allow-cross-machine-baseline")
        if baseline.get("role") != args.role:
            raise SystemExit("baseline detector role does not match")
        failures.extend(
            compare_detector_reports(
                baseline,
                report,
                BenchmarkGate(
                    maximum_p95_regression_percent=args.maximum_p95_regression_percent,
                    minimum_throughput_ratio=args.minimum_throughput_ratio,
                ),
            )
        )
    report["gate"] = {"passed": not failures, "failures": failures}
    if args.output is not None:
        _write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
