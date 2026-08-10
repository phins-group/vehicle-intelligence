#!/usr/bin/env python3
"""Measure Phase 1 component latency on real video/model inputs."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.infrastructure.vision.bytetrack import ByteTrackVehicleTracker
from vehicle_intelligence.infrastructure.vision.factory import (
    create_plate_detector,
    create_vehicle_detector,
)
from vehicle_intelligence.infrastructure.vision.opencv import (
    AdaptivePlatePreprocessor,
    OpenCVVideoSource,
)
from vehicle_intelligence.infrastructure.vision.paddleocr import PaddleOCRProvider


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Benchmark real Phase 1 providers")
    result.add_argument("video", type=Path)
    result.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    result.add_argument("--vehicle-model")
    result.add_argument("--plate-model", required=True)
    result.add_argument("--device")
    result.add_argument("--max-frames", type=int, default=200)
    return result


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    samples = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "meanMs": round(float(samples.mean()), 3),
        "p50Ms": round(float(np.percentile(samples, 50)), 3),
        "p95Ms": round(float(np.percentile(samples, 95)), 3),
        "maxMs": round(float(samples.max()), 3),
    }


def main() -> None:
    args = parser().parse_args()
    if args.max_frames < 1:
        raise SystemExit("--max-frames must be positive")
    settings = load_settings(args.config)
    vehicle_config = settings.vision.vehicle_detection.model_copy(
        update={
            "model_path": args.vehicle_model or settings.vision.vehicle_detection.model_path,
            "device": args.device,
        }
    )
    plate_config = settings.vision.plate_detection.model_copy(
        update={"model_path": args.plate_model, "device": args.device}
    )
    source = OpenCVVideoSource(
        args.video,
        settings.camera.id,
        settings.camera.fps_limit,
    )
    vehicle_detector = create_vehicle_detector(vehicle_config)
    plate_detector = create_plate_detector(plate_config)
    tracker = ByteTrackVehicleTracker(
        settings.tracking,
        min(source.source_fps, settings.camera.fps_limit),
    )
    quality_evaluator = PlateQualityEvaluator(settings.vision.plate_quality)
    preprocessor = AdaptivePlatePreprocessor(settings.vision.preprocessing)
    ocr = PaddleOCRProvider(settings.vision.ocr)
    timings: dict[str, list[float]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    wall_start = time.perf_counter()
    try:
        for index, frame in enumerate(source.frames()):
            if index >= args.max_frames:
                break
            start = time.perf_counter()
            vehicles = vehicle_detector.detect(frame.image)
            timings["vehicleInference"].append(elapsed_ms(start))
            counts["sampledFrames"] += 1
            counts["vehicleDetections"] += len(vehicles)

            start = time.perf_counter()
            tracker.update(vehicles, frame.image)
            timings["tracking"].append(elapsed_ms(start))

            for vehicle in vehicles:
                box = vehicle.bbox.clip(frame.image.shape[1], frame.image.shape[0])
                if box is None:
                    continue
                vehicle_crop = frame.image[box.y1 : box.y2, box.x1 : box.x2]
                start = time.perf_counter()
                plates = plate_detector.detect(vehicle_crop)
                timings["plateInference"].append(elapsed_ms(start))
                counts["plateDetections"] += len(plates)
                for plate in plates:
                    plate_box = plate.bbox.clip(vehicle_crop.shape[1], vehicle_crop.shape[0])
                    if plate_box is None:
                        continue
                    plate_crop = vehicle_crop[
                        plate_box.y1 : plate_box.y2, plate_box.x1 : plate_box.x2
                    ]
                    quality = quality_evaluator.evaluate(plate_crop, plate)
                    if not quality.eligible:
                        counts["qualityRejections"] += 1
                        continue
                    variants = preprocessor.variants(plate_crop, quality, plate)
                    start = time.perf_counter()
                    ocr.recognize(variants[0].image)
                    timings["ocrInference"].append(elapsed_ms(start))
                    counts["ocrRequests"] += 1
    finally:
        source.close()
        tracker.reset()
    wall_seconds = time.perf_counter() - wall_start
    report = {
        "video": str(args.video.resolve()),
        "wallSeconds": round(wall_seconds, 3),
        "effectiveSampledFps": round(counts["sampledFrames"] / max(wall_seconds, 1e-9), 3),
        "counts": dict(counts),
        "latency": {name: summary(values) for name, values in timings.items()},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
