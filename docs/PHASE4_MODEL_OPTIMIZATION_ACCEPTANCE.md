# Phase 4 Model Optimization Acceptance

Accepted on 2026-08-10.

## Implemented

- Config-driven Ultralytics, ONNX Runtime, and TensorRT-EP detector factory.
- SHA-256 verified/versioned model artifacts and export manifests.
- YOLO raw/post-NMS decode, deterministic letterbox, class filtering, NMS, and
  source-coordinate restoration for vehicle and axis-aligned plate detection.
- Explicit CPU/CoreML/CUDA/TensorRT provider selection with fail-closed
  accelerator requirements.
- Component benchmark reports with p50/p95/p99, FPS, load/RSS, provider/model
  identity, absolute thresholds, and baseline regression gates.
- Real ONNX export and runtime-load validation for YOLO11n.

## Tested

- Full real-service Python suite: 183 tests passed with MongoDB, Redis, and MinIO.
- Ruff passed for source, tests, and scripts.
- ONNX 1.22.0 checker and ONNX Runtime 1.23.2 loaded the exported graph.
- Real parity image: PyTorch and ONNX/CoreML each retained one vehicle.
- Development-host result: p95 improved from 82.35 ms to 10.39 ms and effective
  throughput from 12.66 to 97.80 FPS on the parity image.
- Installed runtime exposed CoreML/CPU only; a TensorRT request produced the
  expected explicit unavailable-provider failure.

## Known limitations

- No licensed/evaluated Vietnamese plate checkpoint exists in this workspace;
  plate performance and accuracy were not fabricated.
- ONNX OBB corner decoding is not yet accepted; OBB plate checkpoints retain the
  existing Ultralytics provider.
- Measured results are local-host baselines, not production capacity promises.

## Next

- Add fair latest-frame GPU scheduling, edge packaging, and deployment-level
  capacity benchmarks.
