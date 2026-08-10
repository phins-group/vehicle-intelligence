# Model Runtime Optimization

## Boundary

Detector selection is configuration-driven through the existing
`VehicleDetector` and `PlateDetector` ports. Supported providers are
`yolo`/`ultralytics`, `picodet`, `onnxruntime`, and `tensorrt`. The last value
means ONNX Runtime's TensorRT execution provider; it does not introduce
TensorRT into domain or pipeline code.

Optimized artifacts are SHA-256 checked before a session is created. Runtime
metadata retains the model name, version, and resolved artifact hash. Requested
accelerators are mandatory: an unavailable TensorRT/CUDA/CoreML provider raises
an explicit dependency error instead of silently reporting a CPU benchmark as an
accelerated run. Safe CPU fallback is registered only after the requested
provider successfully exists.

## ONNX detector contract

The adapter performs deterministic YOLO letterbox preprocessing, RGB/NCHW
conversion, confidence filtering, class-aware NMS, and source-coordinate
restoration. It supports current Ultralytics detection exports in either raw
`[1, 4+classes, candidates]` form or post-NMS `[1, candidates, 6]` form.
`onnx_output_format` can remove an ambiguous auto-detection case, and
`model_classes` supplies the exact output mapping for non-COCO vehicle models.

The optimized plate adapter currently supports axis-aligned detection exports.
An OBB model stays on the Ultralytics adapter until an evaluated ONNX OBB decoder
is added; perspective corners are never fabricated.

## PicoDet detector contract

The PicoDet provider uses one checksum-verified ONNX artifact and the same ONNX
execution-provider policy. It implements PaddleDetection's RGB resize,
normalization and NCHW input contract, including optional `scale_factor` and
`im_shape` inputs. It accepts either PaddleDetection post-NMS rows in
`[class, score, x1, y1, x2, y2]` order or raw score/distribution heads. Raw-head
DFL decoding, configured stride validation and class-aware NMS stay inside the
provider. Coordinates are restored and clamped to the exact image supplied by
the caller.

For a vehicle PicoDet model, `model_classes` is mandatory because raw class IDs
are meaningless without the export's ordered label map. A dedicated one-class
plate model defaults to `license_plate`, but an explicit mapping is preferred.
The defaults expose stride, top-k, mean/std and scale under each detector's own
`picodet` block, so vehicle and plate preprocessing can differ safely.

The adapter follows the official PaddleDetection deployment output contract and
PicoDet post-processing algorithm:
[Python inference](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/deploy/python/infer.py) and
[PicoDet post-processing](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.8.1/deploy/python/picodet_postprocess.py).

## Export

Install the optional runtime and export a pinned real checkpoint:

```bash
python -m pip install -e '.[vision,optimization]'
python scripts/export_detector_model.py models/vehicle.pt \
  --format onnx \
  --model-name vehicle-detector \
  --model-version 2026.08 \
  --manifest output/benchmarks/vehicle.onnx.manifest.json
```

The command verifies an optional source checksum, asks Ultralytics to export the
real graph, runs the ONNX checker, loads it with ONNX Runtime, and atomically
writes a version/hash manifest. TensorRT engine export requires an explicit CUDA
device and fails on CPU-only hosts.

## Benchmark and gates

`scripts/benchmark_detector.py` records load time, p50/p95/p99/max latency,
effective FPS, peak RSS, detection-count stability, machine identity, artifact
hash, and actual execution-provider order. A deterministic synthetic image is
valid for repeatable latency only and is explicitly marked as not accuracy
representative. Use a labeled/representative image for output parity.

```bash
python scripts/benchmark_detector.py \
  --model models/vehicle.onnx \
  --provider onnxruntime --execution-provider cuda \
  --role vehicle --model-version 2026.08 \
  --image datasets/benchmark/gate-day.jpg \
  --baseline output/benchmarks/vehicle-pytorch.json \
  --maximum-p95-regression-percent 10 \
  --minimum-throughput-ratio 0.95 \
  --output output/benchmarks/vehicle-onnx.json
```

The command exits non-zero when an absolute or baseline gate fails.

## Measured on the development host

The same YOLO11n checkpoint, 640 input, and vehicle-class filter were measured
on 2026-08-10. These are host-specific engineering results, not production SLA.

| Runtime | Input | p95 | FPS | Detections |
|---|---:|---:|---:|---:|
| Ultralytics/PyTorch CPU | deterministic 1920x1080 | 69.10 ms | 15.22 | 0 |
| ONNX Runtime CPU | deterministic 1920x1080 | 38.97 ms | 25.94 | 0 |
| ONNX Runtime CoreML | deterministic 1920x1080 | 10.09 ms | 100.08 | 0 |
| Ultralytics/PyTorch CPU | Ultralytics bus image | 82.35 ms | 12.66 | 1 |
| ONNX Runtime CoreML | Ultralytics bus image | 10.39 ms | 97.80 | 1 |

The parity image produced one retained vehicle in both runtimes. This host exposes
CoreML/CPU but not CUDA/TensorRT; the TensorRT request was verified to fail
closed. A Vietnamese plate checkpoint is not present in the repository, so no
plate latency or accuracy result is claimed.
