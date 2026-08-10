# Phase 2 Camera Management Acceptance Record

## Scope

This milestone adds central camera configuration and process lifecycle without
moving inference into FastAPI. It accepts camera CRUD, encrypted MongoDB storage,
bounded connection testing, persisted latest health, and desired-state
reconciliation across isolated camera workers.

## Implemented behavior

- Versioned, immutable `Camera` and redacting `SecretUri` domain values.
- Repository/cipher/tester/worker-launcher ports independent of MongoDB, OpenCV,
  cryptography, and subprocess SDK details.
- Camera API for CRUD, enable/disable, connection test, and latest health.
- Optimistic `revision` conflicts prevent lost updates.
- AES-256-GCM credential tokens use a fresh 12-byte nonce and camera-bound AAD.
- Public responses expose `credentialsConfigured` but no RTSP URL or ciphertext.
- Sanitized validation errors cannot echo rejected RTSP credentials.
- MongoDB keeps one latest `camera_health` document per camera.
- A throttled worker reporter persists source FPS, decode FPS, queue/drops,
  reconnect/failure counters, epoch, timestamps, and terminal state.
- The supervisor starts one subprocess per enabled camera, stops disabled/deleted
  cameras, restarts changed revisions, isolates crashes, and applies retry
  backoff. Secrets are absent from child arguments and the master key is absent
  from the child environment.

## Verification — 2026-08-09

The self-contained suite covers the domain value, authenticated encryption,
camera service, API redaction and fail-closed behavior, health throttling,
connection tester, subprocess secret boundary, and multi-camera supervisor.
The opt-in MongoDB integration test additionally inspects the raw stored document,
verifies ciphertext-only persistence, round-trip decryption, fresh ciphertext on
update, optimistic replacement, indexes, and one-document health upserts.

Commands and results:

```text
.venv/bin/python -m pytest -ra
60 passed, 5 skipped in 24.49s

TEST_MONGODB_URI=mongodb://127.0.0.1:27017 \
TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
TEST_MINIO_ENDPOINT=127.0.0.1:9000 \
.venv/bin/python -m pytest -ra
65 passed in 4.40s

.venv/bin/ruff check src tests run_pipeline.py run_camera.py \
  run_event_worker.py run_camera_supervisor.py
All checks passed

.venv/bin/python -m compileall -q src tests run_pipeline.py run_camera.py \
  run_event_worker.py run_camera_supervisor.py
passed

docker compose config --quiet
passed

docker compose build api
passed
```

The packaged supervisor also completed a real one-pass reconciliation against an
empty MongoDB database with no starts, crashes, or failures. A Compose API smoke
test reported `cameraManagement=available`; a temporary disabled camera returned
no credential, and direct raw-document inspection showed only
`rtspUrlEncrypted`, token version/key ID `v1.primary`, and no plaintext marker.
The rebuilt image was also started with a deliberately malformed key: the API
remained healthy while reporting `cameraManagement=unavailable`, confirming that
the camera boundary fails closed without taking event queries down. All
acceptance records were then removed while persistent service volumes were left
intact.

The suite emits one upstream Starlette `TestClient` deprecation warning about
the `httpx` compatibility package. It does not affect test results or production
runtime.

## Not accepted by this milestone

Authentication/RBAC and actor audit were subsequently accepted in
[PHASE2_SECURITY_ACCEPTANCE.md](PHASE2_SECURITY_ACCEPTANCE.md); they were not part
of this earlier camera-management acceptance run.

- ONVIF discovery/configuration.
- Authentication, RBAC, and camera-mutation audit records were accepted later in
  the separate security milestone; they were not exercised by this earlier run.
- Automated credential-key rotation or external secret-manager integration.
- Prometheus/OpenTelemetry export, historical metric aggregation, or alerting.
- Shared-GPU batching/fairness, Angular camera UI, or production soak/load tests.
