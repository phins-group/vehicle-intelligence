# Phase 4 Edge Scheduler Acceptance

Accepted on 2026-08-10.

## Implemented

- Thread-safe bounded per-camera latest-frame queues.
- Ready-camera round-robin fairness, stale rejection, drop-oldest, camera
  capacity/unregister, bounded batch wait, and operational snapshots.
- Batch detector port/coordinator and real Ultralytics multi-image inference.
- Paced concurrent capacity benchmark with Jain fairness, drops, batching,
  throughput, end-to-end tail latency, and threshold exits.
- Strict vehicle/plate ONNX edge manifest generator/validator with SHA-256,
  byte-size, path containment, provider availability, and config/model version.
- Non-root ONNX/Paddle/ByteTrack/MinIO edge image and standalone Compose profile.

## Tested

- Full real-service Python suite: 191 tests passed with MongoDB, Redis, and MinIO.
- Ruff passed for source, tests, and scripts.
- ONNX/CoreML capacity run: 8 cameras × 6 FPS, 144/144 emitted, no drops,
  fairness 1.0, p95 end-to-end 88.84 ms.
- PyTorch CPU accepted run: 4 cameras × 3 FPS, 36/36 emitted, no drops,
  fairness 1.0, p95 247.65 ms.
- Overload run remained bounded and fair: fairness 1.0, 55.56% old frames
  intentionally replaced, pending queue returned to zero.
- Edge Compose configuration rendered successfully.
- Edge image built on ARM64, runs as UID 999, imports PaddleOCR/PaddlePaddle,
  Supervision and ONNX Runtime, and exposes CPU execution provider.
- Missing-manifest container startup failed closed with exit code 1.

## Known limitations

- The workspace still has no Vietnamese plate detector artifact, so a complete
  live RTSP edge container was not falsely started with a vehicle model posing as
  a plate model.
- NVIDIA CUDA/TensorRT image/runtime validation requires the target NVIDIA node;
  this ARM64 Docker host exposes CPU only inside Linux containers.
- Shared scheduling is an explicit worker composition component; existing
  per-camera subprocess isolation remains the safe default.

## Next

- Complete quality metrics, dataset export/retraining loop, UI quality view, and
  final platform acceptance.
