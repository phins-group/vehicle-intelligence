# Fair Scheduling and Edge Deployment

## Shared-device scheduler

`FairLatestFrameScheduler` is a thread-safe application component with one
bounded queue per camera and a ready-camera round robin. It enforces:

- configured camera capacity;
- drop-oldest behavior when one camera outruns inference;
- stale-frame rejection by monotonic age;
- at most one frame per ready camera before that camera returns to the tail;
- bounded batch size and batch-accumulation wait;
- submitted/emitted/oldest-drop/stale-drop/pending and per-camera counters.

`FairInferenceCoordinator` drains a batch and uses `BatchVehicleDetector` when
the selected provider exposes it. Ultralytics now performs real multi-image
inference through that contract. Scalar providers are executed in fair batch
order without being mislabeled as device batching. The existing isolated camera
worker remains the default until a deployment deliberately chooses a shared
device process; enabling the config alone does not move inference into FastAPI.

## Capacity benchmark

The benchmark runs a paced producer concurrently with real inference so decoders
continue to submit while the device is busy:

```bash
python scripts/benchmark_edge_capacity.py \
  --model models/vehicle.onnx \
  --provider onnxruntime --execution-provider cuda \
  --model-version 2026.08 \
  --image datasets/benchmark/gate.jpg \
  --cameras 8 --camera-fps 6 --duration-seconds 60 \
  --batch-size 8 --maximum-frame-age-ms 250 \
  --minimum-fairness 0.98 --maximum-drop-ratio 0.02 \
  --maximum-p95-latency-ms 250 \
  --output output/benchmarks/edge-capacity.json
```

The JSON report includes Jain fairness, every camera's emitted count, drop
reasons, mean batch size, effective FPS, and end-to-end p95. It exits non-zero
when a gate fails.

Development-host measurement with the real YOLO11n parity image:

| Runtime/load | Offered | Emitted | Drop | Fairness | p95 |
|---|---:|---:|---:|---:|---:|
| ONNX/CoreML, 8 cameras × 6 FPS | 144 | 144 | 0% | 1.000 | 88.84 ms |
| PyTorch CPU, 4 cameras × 3 FPS | 36 | 36 | 0% | 1.000 | 247.65 ms |
| PyTorch CPU overload, 8 × 6 FPS | 144 | 64 | 55.56% oldest | 1.000 | 592.10 ms |

The overload result demonstrates intentional recent-frame degradation without
unbounded memory or camera starvation; it is not an accepted capacity target.

## Immutable model manifest

The optimized edge image accepts ONNX Runtime/TensorRT-EP artifacts only. A
manifest must contain exactly one vehicle and one Vietnamese plate detector,
each with relative path, provider, execution-provider request, model name,
version, byte size, and SHA-256. Startup validates a bounded JSON document,
portable path containment, exact bytes/hash, file format, and runtime provider
availability before composing detector environment variables.

Create it from real artifacts:

```bash
python scripts/create_edge_manifest.py \
  --node-id edge-gate-01 --config-version 2026.08.10 \
  --model-root models --output edge-manifest.json \
  --vehicle-model models/vehicle.onnx \
  --vehicle-provider onnxruntime \
  --vehicle-model-name vehicle-detector --vehicle-model-version 2026.08 \
  --vehicle-execution-provider cuda \
  --plate-model models/vietnam-plate.onnx \
  --plate-provider onnxruntime \
  --plate-model-name vietnam-plate --plate-model-version 2026.08 \
  --plate-execution-provider cuda
```

## Container deployment

`Dockerfile.edge` is an ONNX-focused, non-root Python 3.12 image with ByteTrack,
PaddleOCR/PaddlePaddle, ONNX Runtime, MinIO, and required OpenCV system libraries.
Models are never copied into the image; `/models` is read-only and `/data` is a
separate volume. `vehicle-edge-entrypoint` verifies the manifest before replacing
itself with the camera worker.

```bash
EDGE_MANIFEST_PATH=$PWD/edge-manifest.json \
EDGE_MODEL_DIRECTORY=$PWD/models \
EDGE_RTSP_URL='rtsp://...' \
docker compose -f docker-compose.edge.yml --profile edge up -d vision-edge
```

RTSP and MinIO/Redis credentials stay in environment/secret injection and never
enter the command or manifest. The reference image is CPU/ARM64-capable. NVIDIA
deployments must build with the site's certified ONNX Runtime GPU package and
CUDA/TensorRT base/runtime libraries, then run the same provider fail-closed and
capacity gates on that exact node.
