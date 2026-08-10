# Phase 4 Final Platform Acceptance

Accepted on 2026-08-10.

All planned implementation milestones in `ROADMAP.md` are accepted. Remaining
milestones: **0**.

## Implemented

- Bounded, server-aggregated model-quality reporting with explicit denominators,
  UTC windows, daily/model-version slices, feedback state, RBAC, and no-store
  responses.
- Lazy Angular `/model-quality` console plus system-health integration.
- Versioned dataset-sample export lifecycle with atomic Mongo claims, stale-lease
  recovery, bounded retries, retention pins, and idempotent reconciliation.
- Immutable OCR feedback exports with camera-grouped deterministic splits,
  byte/pixel-bounded media reads, safe paths, atomic installation, SHA-256 and
  byte-size verification, model lineage, and machine-readable failure reasons.
- Offline OCR evaluation for exact plate accuracy, micro character accuracy,
  expected calibration error, split/model slices, and validated release gates.
- A final performance-evidence verifier that rejects missing hashes, unmeasured
  gates, non-representative detector inputs, failed artifacts, or an incomplete
  detector/edge evidence set.
- Config, indexes, API/CLI wiring, documentation, ADR, unit/integration tests, and
  the final non-root edge/API/Web images.
- A secret-safe static production-readiness gate with deterministic exit codes,
  model artifact/version/hash verification, and atomic JSON report output.

## Tested

- Python real-service acceptance suite: **203 passed** against MongoDB, Redis,
  and MinIO. One upstream Starlette/httpx deprecation warning remains.
- Ruff passed across source, tests, scripts, dataset-export, and
  production-readiness launchers.
- Angular: TypeScript application/spec checks passed, **9 test files / 33 tests**
  passed, production build passed, and production dependency audit reported
  **0 vulnerabilities**.
- Rebuilt API and Web Compose images. `/api/system/health`, bounded
  `/api/model-quality`, and `/model-quality` returned HTTP 200; the web response
  includes CSP, frame denial, MIME protection, and `Cache-Control: no-store`.
- Rebuilt ARM64 edge image from the accepted source. It runs as UID **999** and
  imports PaddlePaddle 3.2.2, PaddleOCR 3.7.0, Supervision 0.27.0.post2, and ONNX
  Runtime 1.28.0. A missing edge manifest failed closed with exit code 1.
- Final evidence verification accepted **9/9** persisted detector-runtime and
  edge-capacity reports with no failures.
- Dataset-export CLI initialized against live MongoDB and completed the empty
  queue path deterministically with 0 exported / 0 failed.
- The installed `vehicle-production-readiness` entrypoint generated
  `output/production-readiness.json` and failed closed with exit code 4 against
  the deliberate development defaults (18 failures, 8 warnings, 3 passes).

### Measured performance evidence

| Evidence | Result |
| --- | --- |
| ONNX Runtime/CoreML detector parity image | 1 retained vehicle; 97.80 effective FPS; p95 10.385 ms |
| Ultralytics/PyTorch detector parity image | 1 retained vehicle; 12.655 effective FPS; p95 82.348 ms |
| ONNX Runtime/CoreML, 8 cameras × 6 FPS | 144/144 emitted; 0 drops; Jain fairness 1.0; p95 88.842 ms |
| PyTorch CPU, 4 cameras × 3 FPS | 36/36 emitted; 0 drops; Jain fairness 1.0; p95 247.653 ms |
| PyTorch overload, 8 cameras × 6 FPS | bounded drop-oldest 55.56%; Jain fairness 1.0; p95 592.097 ms |

The reports are development-host baselines, not production capacity promises.
The machine-readable summary is
`output/benchmarks/final-performance-gates.json`.

## Acceptance scope

This final milestone closes the planned engineering roadmap: video/RTSP ingress,
tracking and temporal ANPR, event/storage/API paths, camera operations, policy and
security, realtime/UI, observability/retention, production failure validation,
multi-camera identity/topology/ReID/journeys, optimized runtimes, fair scheduling,
edge packaging, and the measured feedback/retraining handoff.

The platform does not automatically train or promote a model. Promotion remains
an explicit offline decision requiring a new immutable model version/hash,
representative holdout results, runtime regression gates, and target-edge
capacity evidence.

## Known limitations

- The workspace does not contain a licensed/evaluated Vietnamese plate-detector
  checkpoint or a representative `sample.mp4`; complete live ANPR accuracy was
  therefore not fabricated. Provider interfaces and executable pipeline paths
  are ready for those artifacts.
- Human-review feedback is intentionally selection-biased. It is regression and
  retraining evidence, not a substitute for a separately curated representative
  holdout.
- TensorRT/CUDA runtime acceptance requires the target NVIDIA node. This ARM64
  Docker host exposes CPU/Azure execution providers inside the Linux edge image.
- No connected in-app browser session was available for pixel-level visual QA.
  Angular type/tests/build and live Nginx/API route smoke were completed instead.
- Production rollout still requires site-specific cameras, topology, credentials,
  retention policy, model thresholds, backup/restore rehearsal, and capacity
  gates on the actual edge hardware.

## Next

- Supply checksum-pinned Vietnamese plate detection/OCR artifacts and a governed,
  representative gate dataset.
- Run the documented end-to-end video and RTSP acceptance on target cameras, then
  establish site-specific quality and capacity baselines before rollout.
