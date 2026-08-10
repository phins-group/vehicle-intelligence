#!/usr/bin/env python3
"""Run a paced multi-camera scheduler benchmark against a real detector."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from vehicle_intelligence.application.benchmarking import latency_summary
from vehicle_intelligence.application.gpu_scheduler import (
    FairInferenceCoordinator,
    FairLatestFrameScheduler,
)
from vehicle_intelligence.application.ports import BatchVehicleDetector
from vehicle_intelligence.config import GPUSchedulerConfig, load_settings
from vehicle_intelligence.domain import VideoFrame
from vehicle_intelligence.infrastructure.vision.factory import create_vehicle_detector
from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark shared-device camera fairness")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--provider",
        choices=("yolo", "ultralytics", "picodet", "onnxruntime", "tensorrt"),
        required=True,
    )
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--device")
    parser.add_argument("--execution-provider", action="append", default=[])
    parser.add_argument("--output-format", choices=("auto", "raw", "nms"), default="auto")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--cameras", type=int, default=8)
    parser.add_argument("--camera-fps", type=float, default=6.0)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--per-camera-queue", type=int, default=1)
    parser.add_argument("--maximum-frame-age-ms", type=float, default=250.0)
    parser.add_argument("--batch-wait-ms", type=float, default=5.0)
    parser.add_argument("--minimum-fairness", type=float, default=0.95)
    parser.add_argument("--maximum-drop-ratio", type=float, default=0.10)
    parser.add_argument("--maximum-p95-latency-ms", type=float, default=250.0)
    parser.add_argument("--output", type=Path)
    return parser


def _input_image(path: Path | None) -> tuple[np.ndarray, str]:
    if path is not None:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"cannot decode benchmark image: {path}")
        return image, str(path.expanduser().resolve())
    y, x = np.indices((720, 1280), dtype=np.uint16)
    return (
        np.stack((x % 256, y % 256, (x + y) % 256), axis=2).astype(np.uint8),
        "deterministic-synthetic",
    )


def _jain_fairness(counts: list[int]) -> float:
    total = sum(counts)
    square_sum = sum(value * value for value in counts)
    if not counts or square_sum == 0:
        return 0.0
    return total * total / (len(counts) * square_sum)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.cameras <= 1024:
        raise SystemExit("camera count must be in [1, 1024]")
    if not 0 < args.camera_fps <= 120 or not 0 < args.duration_seconds <= 3600:
        raise SystemExit("camera FPS/duration are outside benchmark bounds")
    if not 0 < args.minimum_fairness <= 1 or not 0 <= args.maximum_drop_ratio <= 1:
        raise SystemExit("invalid benchmark gates")
    if args.warmup_batches < 0:
        raise SystemExit("warmup batch count cannot be negative")
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise SystemExit(f"model artifact does not exist: {model_path}")
    settings = load_settings(args.config)
    detector_config = settings.vision.vehicle_detection.model_copy(
        update={
            "provider": args.provider,
            "model_path": str(model_path),
            "model_version": args.model_version,
            "device": args.device,
            "execution_providers": args.execution_provider,
            "onnx_output_format": args.output_format,
        }
    )
    detector = create_vehicle_detector(detector_config)
    scheduler_config = GPUSchedulerConfig(
        enabled=True,
        maximum_cameras=args.cameras,
        maximum_batch_size=args.batch_size,
        per_camera_queue_size=args.per_camera_queue,
        maximum_frame_age_ms=args.maximum_frame_age_ms,
        batch_wait_ms=args.batch_wait_ms,
    )
    scheduler = FairLatestFrameScheduler(scheduler_config)
    coordinator = FairInferenceCoordinator(scheduler, detector)
    image, input_name = _input_image(args.image)
    warmup_size = min(args.batch_size, args.cameras)
    for _ in range(args.warmup_batches):
        if isinstance(detector, BatchVehicleDetector):
            detector.detect_batch([image] * warmup_size)
        else:
            for _ in range(warmup_size):
                detector.detect(image)
    camera_ids = [f"benchmark-camera-{index:03d}" for index in range(args.cameras)]
    production_done = threading.Event()
    producer_error: list[BaseException] = []

    def produce() -> None:
        try:
            interval = 1 / args.camera_fps
            started = time.monotonic()
            deadline = started + args.duration_seconds
            next_at = [started] * args.cameras
            frame_ids = [0] * args.cameras
            while time.monotonic() < deadline:
                now = time.monotonic()
                for index, camera_id in enumerate(camera_ids):
                    while next_at[index] <= now and next_at[index] < deadline:
                        scheduler.submit(
                            VideoFrame(
                                camera_id=camera_id,
                                frame_id=frame_ids[index],
                                timestamp=datetime.now(UTC),
                                image=image,
                            ),
                            now_monotonic=now,
                        )
                        frame_ids[index] += 1
                        next_at[index] += interval
                nearest = min(next_at)
                time.sleep(max(0, min(0.002, nearest - time.monotonic())))
        except BaseException as exc:
            producer_error.append(exc)
        finally:
            production_done.set()

    producer = threading.Thread(target=produce, name="benchmark-frame-producer", daemon=True)
    measured_started = time.monotonic()
    producer.start()
    latencies: list[float] = []
    batch_sizes: list[int] = []
    while not production_done.is_set() or scheduler.snapshot().pending:
        results = coordinator.run_once(wait_seconds=0.05)
        if results:
            batch_sizes.append(len(results))
            latencies.extend(item.end_to_end_latency_ms for item in results)
    producer.join(timeout=1)
    if producer_error:
        raise producer_error[0]
    elapsed = time.monotonic() - measured_started
    snapshot = scheduler.snapshot()
    counts = [snapshot.emitted_per_camera.get(camera_id, 0) for camera_id in camera_ids]
    fairness = _jain_fairness(counts)
    dropped = snapshot.dropped_oldest + snapshot.dropped_stale
    drop_ratio = dropped / max(snapshot.submitted, 1)
    latency = latency_summary(latencies)
    p95_latency = float(latency.get("p95Ms", math.inf))
    failures: list[str] = []
    if fairness < args.minimum_fairness:
        failures.append(f"fairness {fairness:.4f} is below {args.minimum_fairness:.4f}")
    if drop_ratio > args.maximum_drop_ratio:
        failures.append(f"drop ratio {drop_ratio:.4f} exceeds {args.maximum_drop_ratio:.4f}")
    if p95_latency > args.maximum_p95_latency_ms:
        failures.append(
            f"p95 end-to-end latency {p95_latency:.3f}ms exceeds "
            f"{args.maximum_p95_latency_ms:.3f}ms"
        )
    report: dict[str, object] = {
        "schemaVersion": 1,
        "kind": "edge-capacity-benchmark",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": {
            "path": str(model_path),
            "sha256": sha256_file(model_path),
            "version": args.model_version,
            "provider": args.provider,
            "executionProviders": list(
                getattr(detector, "execution_providers", (args.device or "auto",))
            ),
        },
        "input": input_name,
        "offered": {
            "cameras": args.cameras,
            "fpsPerCamera": args.camera_fps,
            "durationSeconds": args.duration_seconds,
            "frames": snapshot.submitted,
        },
        "scheduler": {
            "maximumBatchSize": args.batch_size,
            "perCameraQueueSize": args.per_camera_queue,
            "maximumFrameAgeMs": args.maximum_frame_age_ms,
            "batchWaitMs": args.batch_wait_ms,
            "warmupBatches": args.warmup_batches,
            "emittedFrames": snapshot.emitted,
            "droppedOldest": snapshot.dropped_oldest,
            "droppedStale": snapshot.dropped_stale,
            "dropRatio": round(drop_ratio, 6),
            "meanBatchSize": round(sum(batch_sizes) / max(len(batch_sizes), 1), 3),
            "inferenceBatches": len(batch_sizes),
            "jainFairness": round(fairness, 6),
            "perCameraEmitted": snapshot.emitted_per_camera,
        },
        "result": {
            "elapsedSeconds": round(elapsed, 3),
            "effectiveFps": round(snapshot.emitted / max(elapsed, 1e-9), 3),
            "endToEndLatency": latency,
        },
        "gate": {"passed": not failures, "failures": failures},
    }
    if args.output is not None:
        _atomic_json(args.output.expanduser().resolve(), report)
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
