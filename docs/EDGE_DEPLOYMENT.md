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

When `gpu_scheduler.enabled=true`, `vehicle-camera-supervisor` starts one dedicated
`vehicle-inference-service` before it starts camera workers. That process loads the
vehicle and plate models once and exposes bounded, length-prefixed binary requests
over a mode-0600 Unix socket. Camera-bound HMAC capabilities are passed through
one-shot inherited file descriptors; raw images never use temporary files or
base64. Requests are admitted by global/per-camera byte and call budgets, then
scheduled one image per ready camera before a camera returns to the round-robin
tail. Providers with `detect_batch` receive real multi-image batches; scalar
providers retain the same result mapping.

The camera subprocess path remains unchanged when the scheduler is disabled.
When enabled, workers fail closed on socket/authentication/deadline errors and do
not load a local detector fallback. A detector call exceeding
`request_timeout_seconds` makes the daemon unhealthy; the supervisor stops camera
workers and applies capped exponential backoff before restarting the daemon and
then the workers. `maximum_frame_age_ms` applies only before a request's first
image is dispatched; an already-started multi-batch request completes under the
end-to-end request timeout.

A fast batch-provider exception is isolated by bounded binary subdivision under
that same request deadline. Successful camera results retain their original
mapping, while only calls containing an isolated bad image fail. Isolation is
safe here because detector inference is local and read-only; it must not be used
around providers with externally visible side effects. Repeated isolated failures
temporarily quarantine only that camera/detector pair before its next image body
is read. Consecutive fully failed batches spanning the configured minimum camera
count open the provider circuit breaker, make the daemon unhealthy, and let the
supervisor restart the model process. `maximum_isolation_attempts` bounds retry
amplification; when exhausted, unresolved calls fail closed without publishing
partial results.

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

`/data` is also the durability boundary for the enabled finalization outbox.
Keep `edge_output` on persistent storage across container replacement; an
ephemeral or read-only `/data` makes the worker fail closed before finalizing a
track. Startup and background replay drain staged event envelopes and JPEGs after
MinIO, Redis, or repository recovery. Capacity is bounded and no queued evidence
is evicted automatically. The configured byte bound applies per camera worker;
provision `/data` for at least the sum of all camera bounds plus operating
headroom, and alert on outbox occupancy and filesystem free space.
The edge service grants 75 seconds for termination so the outbox's 60-second
final drain and the bounded object-storage request tail can finish before Docker
escalates to `SIGKILL`.

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
