# Vehicle Intelligence Platform

Edge-first ANPR and vehicle-event pipeline. This repository implements the
Phase 1 video-file path and the implemented Phase 2 runtime foundation: detection,
ByteTrack, plate quality/OCR, Vietnamese plate normalization, temporal voting,
reconnect-safe track finalization, media storage, Redis Streams event delivery,
MongoDB persistence, encrypted camera management, isolated multi-camera workers,
persisted latest health, watchlists, declarative rules, durable idempotent actions,
alerts, pluggable Bearer authentication, RBAC, append-only audit logs, Redis
Pub/Sub fan-out, authorized SSE/WebSocket delivery, a query/management API, an
Angular operator console, event-scoped signed media presentation, and
revisioned human OCR review with dataset feedback, an optional bounded live
monitor with configurable overlays, Prometheus/OpenTelemetry observability,
coordinated MongoDB/MinIO retention with dataset pins, and the Phase 3 logical
identity/fingerprint foundation with separately versioned embedding storage.
The camera manager also supports bounded credential-free ONVIF discovery,
explicit batch admission outcomes, and capacity/start-rate limited supervision.

The vision layer is provider-based. YOLO/Ultralytics, PicoDet/ONNX Runtime,
PaddleOCR, ByteTrack, MongoDB, and MinIO are infrastructure adapters; domain and
application code do not import their SDKs. Vehicle and plate detector providers
are configured independently.

## Quick start

Python 3.12 is required.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,vision,minio,optimization]'

# PaddleOCR is the provider package; local inference also needs the platform
# engine from PaddlePaddle's official CPU index. Intel macOS uses the last
# available x86_64 wheel:
python -m pip install paddlepaddle==3.0.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

pytest
```

Run a real video after supplying a Vietnamese plate detector checkpoint:

```bash
python run_pipeline.py sample.mp4 \
  --plate-model models/vietnam-plate.pt \
  --vehicle-model yolo11n.pt
```

Detector paths must resolve to local files. Runtime loading does not implicitly
download checkpoints, and any configured `model_hash` is verified before a
model is deserialized.

To bypass vehicle detection and track plates directly on each sampled full
frame, enable plate-only mode. The vehicle model is not loaded:

```bash
python run_pipeline.py sample.mp4 \
  --plate-only \
  --plate-model models/vietnam-plate.pt \
  --device cpu \
  --no-mongo
```

Plate-only events keep the canonical event schema with `vehicle.type=unknown`,
omit the vehicle crop and vehicle-detector trace, and retain plate tracking,
quality filtering, OCR, temporal voting, snapshots, and plate crops. Use a plate
checkpoint evaluated on full camera frames; a detector trained only on tight
vehicle crops may miss small distant plates.

MongoDB is optional. By default the CLI writes `output/events.jsonl` and media
under `output/`. Enable MongoDB with `--mongo` and configuration/environment
values described in [configs/default.yaml](configs/default.yaml).

Start the Phase 1 local infrastructure and API with:

```bash
install -m 600 .env.example .env
# Generate once and place the result in .env as
# VIP_SECURITY__CAMERA_CREDENTIAL_KEY=<value>.
python -c 'import secrets; print(secrets.token_urlsafe(32))'
docker compose up -d mongodb redis minio api
curl http://localhost:8000/api/system/health
curl http://localhost:8000/livez
curl http://localhost:8000/readyz
```

`/livez` is process-only liveness and is used by the image healthcheck. `/readyz`
returns `503` until startup finishes or while the configured canonical MongoDB
event store is unavailable. Optional camera-management, MinIO, realtime, and live
monitor outages are reported as degraded checks without removing the REST API
from service. Keep `/api/system/health` for the backward-compatible capability
summary; ingress and orchestrators should query `/readyz` on the API listener
(port `8000` by default) for traffic admission.

Compose binds every published port to `127.0.0.1` by default. Set
`HOST_BIND_ADDRESS=0.0.0.0` only when remote access is intentional and protected
by the appropriate firewall, TLS, and authentication controls.

The API container runs as the non-root `vehicle` user. On Linux hosts, ensure
the bind-mounted `./datasets` directory is writable by that container user; use
`docker compose run --rm --no-deps --entrypoint sh api -c 'id -u; id -g'` to
inspect its UID/GID before adjusting directory ownership. Avoid making the
dataset directory world-writable.

Start the same-origin Angular operator console with the platform services:

```bash
docker compose up -d mongodb redis api web
open http://localhost:4200
```

The console includes dashboard, vehicle-event search/live updates, exact
plate-centric history, short-lived snapshot/plate/clip evidence, camera
management, alert workflow, OCR correction, watchlist management, structured
rule authoring, low-rate live monitoring, and system-health screens. See
[Angular operator dashboard](docs/WEB_DASHBOARD.md) for local Node setup,
authentication behavior, deployment details, and current screen limits.

The key must decode to exactly 32 bytes. If MongoDB is enabled without a valid
key, the API intentionally keeps event queries alive but reports camera management
as unavailable and returns `503` for camera operations. It never stores a
plaintext RTSP URL or returns either the URL or encrypted token to clients.

For host-native API development, start MongoDB separately and run:

```bash
VIP_MONGODB__ENABLED=true vehicle-api
```

The implemented surface includes event queries, cursor-paginated final-plate
search, human OCR review/dataset samples, camera management, bounded live
monitoring, watchlist/rule management, alert review, and system health.
Event and alert listings use opaque cursors. See [Event contracts](docs/EVENTS.md)
and [Rules and alerts](docs/RULES_AND_ALERTS.md).

The CLI intentionally refuses to run without a plate model. The project does
not ship an unverified checkpoint or fabricate recognition output. See
[model requirements and the optional pinned local-demo checkpoint](models/README.md#local-demo-checkpoint)
for download, checksum, compatibility, and Vietnamese evaluation limitations.

## Offline vehicle and plate model training

Vehicle and plate model training is isolated from the online camera runtime.
The repository now provides group-safe immutable COCO dataset builds,
provider-neutral detector evaluation, PaddleDetection/PicoDet subprocess
orchestration, ONNX candidate packaging, and optional private Hugging Face
dataset/model/Job adapters:

```bash
python -m pip install -e '.[dev,training,optimization]'
python run_model_training.py --config configs/model-training.yaml --help
```

No model binary or claimed accuracy is generated without real labeled data and
a completed training/evaluation run. See the complete source annotation
contract and workflow in [Detector training](docs/DETECTOR_TRAINING.md).
The authenticated operator workflow is available at `/dataset-review`: it
supports visual bbox review, immutable decision revisions, and ADMIN-only
promotion into a new source version without modifying the original dataset.
After promotion, `/datasets` catalogs immutable source/export lineage and lets
ADMIN build, verify, and synchronize the selected export to the configured
private Hugging Face dataset repository. Restricted plate data requires both a
server-side policy opt-in and explicit per-request confirmation; the Hub token
never enters browser state. `/model-training` then pins that reviewed source,
COCO manifest and private Hub commit into an auditable GPU training run, with
preflight blockers, status/log polling and ADMIN-only cancellation. It does not
mark a checkpoint production-ready before ONNX evaluation and release gates. See
[Detector training](docs/DETECTOR_TRAINING.md#build-model-operator-screen).

On Linux or Apple Silicon, select the current PaddlePaddle CPU/GPU command from
the [official installation guide](https://www.paddlepaddle.org.cn/documentation/docs/en/install/index_en.html)
instead of the Intel macOS compatibility command above. Model downloads can be
redirected with `PADDLE_PDX_CACHE_HOME`.

## Single-camera RTSP worker

Run one isolated camera worker after supplying the same verified model files:

```bash
export GATE01_RTSP_URL='rtsp://user:password@camera.example/live'
python run_camera.py \
  --camera gate01 \
  --rtsp-env GATE01_RTSP_URL \
  --plate-model models/vietnam-plate.pt \
  --vehicle-model yolo11n.pt
```

`--rtsp-env` is preferred because it avoids placing credentials in the command
line and shell history. The source URL is held as a secret and is never included
in worker logs or the generated source/track ID.

The RTSP worker accepts the same `--plate-only` switch; when enabled, omit
`--vehicle-model` unless it is retained only as an unused config override.

For a repeatable local RTSP relay, start the optional MediaMTX Compose profile
based on its [official Docker installation guidance](https://mediamtx.org/docs/kickoff/install):

```bash
docker compose --profile rtsp-dev up -d mediamtx

# Run in a second terminal. Stream copy requires H.264-compatible input.
ffmpeg -re -stream_loop -1 -i sample.mp4 -c:v copy -an \
  -f rtsp -rtsp_transport tcp rtsp://127.0.0.1:8554/gate01

# Run in a third terminal.
export GATE01_RTSP_URL='rtsp://127.0.0.1:8554/gate01'
python run_camera.py --camera gate01 --rtsp-env GATE01_RTSP_URL \
  --plate-model models/vietnam-plate.pt
```

The decoder samples at `camera.fps_limit`, keeps only a bounded number of recent
frames, drops stale frames under backpressure, reconnects with capped exponential
backoff, and assigns a new stream epoch after reconnect. `Ctrl-C` finalizes active
tracks before shutdown. See [Camera pipeline](docs/CAMERA_PIPELINE.md) for the
lifecycle and current limitations.

The default JSONL sink is intended for local development. Configure MongoDB and
run with `--mongo` for a long-running/high-volume worker; the current JSONL query
index is memory-backed.

## Central camera manager and supervisor

Camera CRUD is available at `/api/cameras`. A camera document carries an
optimistic `revision`; full `PUT` updates must provide it, and may omit
`stream.rtspUrl` to retain the existing credential. Lifecycle and operations are:

```text
POST   /api/cameras
POST   /api/cameras/batch
POST   /api/cameras/discover
GET    /api/cameras?enabledOnly=true
GET    /api/cameras/{id}
PUT    /api/cameras/{id}
DELETE /api/cameras/{id}
POST   /api/cameras/{id}/enable
POST   /api/cameras/{id}/disable
POST   /api/cameras/{id}/test-connection
GET    /api/cameras/{id}/health
```

Run the supervisor host-native so each enabled camera gets an isolated vision
process and can use the host GPU/model cache:

```bash
export VIP_MONGODB__ENABLED=true
export VIP_MONGODB__URI='mongodb://127.0.0.1:27017'
export VIP_SECURITY__CAMERA_CREDENTIAL_KEY='<32-byte URL-safe base64 value>'
export VIP_REDIS__URL='redis://127.0.0.1:6379/0'
export VIP_LIVE_MONITOR__ENABLED=true
# Optional: load vehicle/plate models once and batch inference across cameras.
export VIP_GPU_SCHEDULER__ENABLED=true
vehicle-camera-supervisor
```

The supervisor continuously reconciles MongoDB configuration. It starts/stops a
worker when a camera is enabled/disabled, restarts it after a revision change or
crash, applies bounded restart backoff, and caps simultaneous workers and starts
per reconciliation cycle. With the GPU scheduler enabled, it starts the bounded
shared inference daemon before any camera worker and restarts dependent workers
only after the daemon is healthy. ONVIF discovery returns temporary credential-free
device metadata only; an ADMIN must explicitly create each camera and provide
its RTSP URL. The RTSP URL is passed only in a child
environment variable—not as an argument—and the child cannot inherit the
credential-encryption key. See [Camera pipeline](docs/CAMERA_PIPELINE.md).

## Redis event worker

`direct` remains the default backend, so local video and camera runs need no
broker. To separate vision from persistence, start Redis, MongoDB, and the event
worker, then select the Redis publisher at the vision CLI:

```bash
docker compose --profile event-driven up -d mongodb redis event-worker

export VIP_REDIS__URL='redis://127.0.0.1:6379/0'
python run_camera.py \
  --camera gate01 \
  --rtsp-env GATE01_RTSP_URL \
  --plate-model models/vietnam-plate.pt \
  --event-backend redis
```

For a host-native event worker:

```bash
VIP_EVENT_BUS__BACKEND=redis \
VIP_MONGODB__ENABLED=true \
VIP_MONGODB__URI=mongodb://127.0.0.1:27017 \
VIP_REDIS__URL=redis://127.0.0.1:6379/0 \
vehicle-event-worker
```

The consumer group uses at-least-once delivery. A message is ACKed only after the
event is durably persisted and every matched rule action has reached a durable
terminal/success state. Stale pending messages are reclaimed; invalid contracts
move to the bounded dead-letter stream. Each Redis batch preserves ordering per
camera while processing different cameras with bounded concurrency, then ACKs
successful messages together. Enabled rules are validated once per short TTL
instead of being reloaded for every event. See [Event contracts](docs/EVENTS.md).

## Watchlists, rules, actions, and alerts

Policy evaluation belongs to the central Redis event worker. It is deliberately
not called from detector/OCR code. A typical blacklist rule is:

```json
{
  "id": "rule-blacklist-alert",
  "name": "Alert on blacklisted vehicle",
  "priority": 100,
  "enabled": true,
  "conditions": [
    {"field": "watchlist", "operator": "CONTAINS", "value": "BLACKLIST"}
  ],
  "actions": [
    {
      "id": "create-alert",
      "type": "CREATE_ALERT",
      "parameters": {"severity": "CRITICAL", "message": "Blacklisted vehicle"}
    }
  ]
}
```

Create the normalized watchlist entry and rule through `/api/watchlists` and
`/api/rules`; review generated alerts at `/api/alerts`. Updates use optimistic
`revision` values. The supported fields, operators, actions, request examples,
retry behavior, and alert lifecycle are in
[Rules and alerts](docs/RULES_AND_ALERTS.md).

`CREATE_ALERT` and `LOG` work with the default safe configuration. Barrier,
webhook, notification, and generic HTTP actions remain disabled until both of
these are explicitly configured:

```bash
VIP_RULE_ENGINE__EXTERNAL_ACTIONS_ENABLED=true
VIP_RULE_ENGINE__EXTERNAL_ALLOWED_HOSTS='["barrier.internal"]'
```

External calls carry a stable `Idempotency-Key`. The receiving system must honor
that key because a process can fail after the remote side effect succeeds but
before the local success record is committed. Redirects and URL-embedded
credentials are rejected.

## Authentication, RBAC, and audit

Local development intentionally defaults to authentication disabled and reports
that state at `/api/system/health`. Before exposing the API, generate a
high-entropy key/verifier pair:

```bash
python scripts/generate_api_key.py
```

Store the raw key in a secret manager and configure only its SHA-256 verifier:

```bash
VIP_AUTH__ENABLED=true
VIP_AUTH__PRINCIPALS='[{"id":"admin-01","display_name":"Platform Admin","role":"ADMIN","key_sha256":"<64-hex-verifier>","enabled":true}]'
```

Clients send `Authorization: Bearer <raw-key>`. `VIEWER` can read; `OPERATOR` can
also test cameras, transition alerts, and review OCR plates; `ADMIN` owns
camera/policy mutations and
audit queries. Successful sensitive actions append redacted before/after records
queryable at `/api/audit-logs`. See
[Security and audit](docs/SECURITY_AND_AUDIT.md) for the complete matrix, key
handling, failure semantics, and current limitations. TLS is mandatory outside
isolated local development.

## Realtime events

With `realtime.enabled=true`, the event worker publishes each durably processed
event to Redis Pub/Sub and every API replica fans it out over:

```text
GET /api/events/stream   # SSE with Authorization header
WS  /ws/events           # Authorization header or browser first-frame auth
GET /api/realtime/health
```

Quick SSE check:

```bash
curl -N \
  -H "Authorization: Bearer ${VEHICLE_API_KEY}" \
  http://localhost:8000/api/events/stream
```

Per-client queues and replay history are bounded. A `system.realtime.gap`
message means the client must reconcile through `/api/events`; Pub/Sub is a
low-latency signal and MongoDB remains canonical history. Raw API keys are not
accepted in query parameters. See [Realtime events](docs/REALTIME_EVENTS.md) for
WebSocket browser authentication, replay, backpressure, and delivery semantics.

## Live monitor

Live monitoring is a separate, optional low-rate path. A camera worker keeps at
most one pending preview, resizes/encodes it in the background, and publishes a
versioned size-capped packet to `vehicle.live.frames`. The API keeps a bounded
ring and exposes authenticated state plus the exact JPEG sequence:

```text
GET /api/cameras/{cameraId}/live
GET /api/cameras/{cameraId}/live/frame?sequence=123
GET /api/live-monitor/health
```

Open `/live-monitor?camera=gate01` in the Angular console to toggle vehicle and
plate boxes, track IDs, plate text, direction, confidence, ROI, and crossing
line. Preview frames are not canonical events, event-WebSocket payloads, or
stored media. See [Live Monitor](docs/LIVE_MONITOR.md) for the contract,
configuration, failure isolation, and multi-replica routing limitation.

## Observability and retention

The API exposes low-cardinality Prometheus metrics on its internal `/metrics`
endpoint. Optional FastAPI OpenTelemetry spans use OTLP/HTTP with bounded
sampling/export timeout and add trace/span IDs to structured logs. Camera-worker
counters and inference latency are persisted as latest `camera_health` state and
rendered centrally without raw event/plate/request IDs as labels.

The maintenance worker coordinates canonical media and event retention through
bounded MongoDB leases. It removes the public media key before object deletion,
restores it after a storage failure, reclaims stale leases, and preserves both
media and source events referenced by `READY` dataset samples. MinIO lifecycle
is deliberately limited to `debug/` and `temporary/`; it does not bypass pins on
canonical `vehicles/` media.

```bash
docker compose --profile observability --profile maintenance --profile event-driven up -d \
  api event-worker prometheus otel-collector retention-worker

# One bounded host-native pass (MongoDB must also be enabled/configured):
VIP_RETENTION__ENABLED=true vehicle-retention-worker --once
```

See [Observability and retention](docs/OBSERVABILITY_AND_RETENTION.md) for the
metric contract, OTLP settings, retention state machine, and operational queries.

## Production identity, transactions, and credential rotation

Production Compose uses a single-node MongoDB replica set and enables atomic
resource plus required-audit transactions. Authentication can remain static API
key or switch to OIDC/JWKS through `VIP_AUTH__PROVIDER`. External actions can use
server-owned Bearer/HMAC credentials and per-target circuit breakers; rule
documents never carry those secrets.

Camera AES-GCM tokens support an active/decrypt keyring. Retain the old key, add
the new key as active, then run:

```bash
vehicle-credential-rotation --dry-run
vehicle-credential-rotation
```

See [Security and audit](docs/SECURITY_AND_AUDIT.md) and the
[production security acceptance](docs/PHASE2_PRODUCTION_SECURITY_ACCEPTANCE.md).

## Production load and soak

The event-path benchmark uses real Redis Streams and MongoDB, injects duplicate
and invalid delivery, enforces thresholds, and cleans its namespaced artifacts:

```bash
python scripts/benchmark_event_path.py --events 5000 \
  --minimum-throughput 100 --maximum-p95-batch-ms 2500
```

Use `--soak-seconds` with `--rate` for paced validation. See
[production validation](docs/PRODUCTION_VALIDATION.md).

## Vehicle identity foundation

The event worker assigns every canonical event a deterministic bootstrap
`vehicleId` and immutable `VehicleFingerprint`. Equal plate strings remain
separate observations until ReID supplies independent evidence; no event array is
embedded into an identity. Visual embeddings are optional, checkpoint-hash
verified, model-versioned, stored outside events/fingerprints, and compared only
over an explicit bounded candidate set.

```text
GET /api/vehicles/{vehicleId}
GET /api/vehicles/{vehicleId}/fingerprints?limit=200
```

See the [identity foundation acceptance](docs/PHASE3_IDENTITY_FOUNDATION_ACCEPTANCE.md)
and [ADR-019](docs/adr/ADR-019-observation-bootstrap-identities-and-bounded-vectors.md).

Directed camera topology is managed through `/api/camera-topology`. Candidate
generation for `/api/vehicle-fingerprints/{id}/candidates` searches only enabled
inbound edges and their indexed travel-time windows, with hard per-edge and total
limits. Missing topology returns no unsafe global fallback.

ReID adds versioned plate/embedding/type/color/travel-time score explanations,
but never auto-merges. OPERATOR/ADMIN identity merge and split are explicit,
revisioned, idempotent, transactional, and audited. See
[Vehicle Re-identification](docs/VEHICLE_REID.md).

Logical vehicle detail and journey APIs derive bounded chronological history from
canonical events, annotate consecutive observations with exact directed topology,
and drive `/vehicles/:vehicleId` in the operator console. See
[Vehicle Journey and Timeline](docs/VEHICLE_JOURNEY.md).

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Domain model](docs/DOMAIN_MODEL.md)
- [Vision pipeline](docs/VISION_PIPELINE.md)
- [Event contracts](docs/EVENTS.md)
- [MongoDB schema](docs/MONGODB_SCHEMA.md)
- [Camera pipeline](docs/CAMERA_PIPELINE.md)
- [Rules and alerts](docs/RULES_AND_ALERTS.md)
- [Security and audit](docs/SECURITY_AND_AUDIT.md)
- [Realtime events](docs/REALTIME_EVENTS.md)
- [Live Monitor](docs/LIVE_MONITOR.md)
- [Observability and retention](docs/OBSERVABILITY_AND_RETENTION.md)
- [Angular operator dashboard](docs/WEB_DASHBOARD.md)
- [Roadmap](docs/ROADMAP.md)
- [Phase 1 acceptance](docs/PHASE1_ACCEPTANCE.md)
- [Phase 2 RTSP acceptance](docs/PHASE2_RTSP_ACCEPTANCE.md)
- [Phase 2 event-bus acceptance](docs/PHASE2_EVENT_BUS_ACCEPTANCE.md)
- [Phase 2 camera-management acceptance](docs/PHASE2_CAMERA_MANAGEMENT_ACCEPTANCE.md)
- [Phase 2 policy-engine acceptance](docs/PHASE2_POLICY_ENGINE_ACCEPTANCE.md)
- [Phase 2 security acceptance](docs/PHASE2_SECURITY_ACCEPTANCE.md)
- [Phase 2 realtime acceptance](docs/PHASE2_REALTIME_ACCEPTANCE.md)
- [Phase 2 operator-dashboard acceptance](docs/PHASE2_OPERATOR_DASHBOARD_ACCEPTANCE.md)
- [Phase 2 policy-console acceptance](docs/PHASE2_POLICY_UI_ACCEPTANCE.md)
- [Phase 2 vehicle-search acceptance](docs/PHASE2_VEHICLE_SEARCH_UI_ACCEPTANCE.md)
- [Phase 2 signed-media acceptance](docs/PHASE2_SIGNED_MEDIA_ACCEPTANCE.md)
- [Phase 2 Live Monitor acceptance](docs/PHASE2_LIVE_MONITOR_ACCEPTANCE.md)
- [Phase 2 ONVIF/multi-camera acceptance](docs/PHASE2_ONVIF_MULTI_CAMERA_ACCEPTANCE.md)
- [Phase 2 observability/retention acceptance](docs/PHASE2_OBSERVABILITY_RETENTION_ACCEPTANCE.md)
- [Phase 2 production-security acceptance](docs/PHASE2_PRODUCTION_SECURITY_ACCEPTANCE.md)
- [Phase 2 production-validation acceptance](docs/PHASE2_PRODUCTION_VALIDATION_ACCEPTANCE.md)
- [Phase 3 identity-foundation acceptance](docs/PHASE3_IDENTITY_FOUNDATION_ACCEPTANCE.md)
- [Phase 3 topology/candidate acceptance](docs/PHASE3_TOPOLOGY_CANDIDATES_ACCEPTANCE.md)
- [Phase 3 ReID/review acceptance](docs/PHASE3_REID_REVIEW_ACCEPTANCE.md)
- [Phase 3 journey/UI acceptance](docs/PHASE3_JOURNEY_UI_ACCEPTANCE.md)
- [Vehicle journey and timeline](docs/VEHICLE_JOURNEY.md)
- [Model runtime optimization](docs/MODEL_OPTIMIZATION.md)
- [Phase 4 model optimization acceptance](docs/PHASE4_MODEL_OPTIMIZATION_ACCEPTANCE.md)
- [Fair scheduling and edge deployment](docs/EDGE_DEPLOYMENT.md)
- [Phase 4 edge scheduler acceptance](docs/PHASE4_EDGE_SCHEDULER_ACCEPTANCE.md)
- [Model quality and retraining feedback](docs/MODEL_QUALITY_AND_RETRAINING.md)
- [PHINS detector dataset governance](docs/PHINS_DATASET_GOVERNANCE.md)
- [Production readiness gate](docs/PRODUCTION_READINESS.md)
- [Final Phase 4/platform acceptance](docs/PHASE4_FINAL_ACCEPTANCE.md)
- [Model requirements](models/README.md)

## Benchmark

Run component latency/FPS measurements against a real video and real weights:

```bash
python scripts/benchmark_pipeline.py sample.mp4 \
  --plate-model models/vietnam-plate.pt \
  --max-frames 200
```

Export and benchmark one detector runtime with machine-readable regression
gates:

```bash
python scripts/export_detector_model.py models/vehicle.pt \
  --format onnx --model-name vehicle-detector --model-version 2026.08
python scripts/benchmark_detector.py \
  --model models/vehicle.onnx --provider onnxruntime \
  --execution-provider cuda --role vehicle --model-version 2026.08 \
  --image datasets/benchmark/gate.jpg \
  --output output/benchmarks/vehicle-onnx.json
```

For multi-camera fairness/capacity and immutable edge packaging, see
[Fair Scheduling and Edge Deployment](docs/EDGE_DEPLOYMENT.md).

Verify the persisted benchmark set before accepting it:

```bash
python scripts/verify_performance_gates.py output/benchmarks
```

The verifier independently enforces the default edge ceilings of 10% dropped
frames and 250 ms p95 latency, so an overload report cannot pass merely because
it was generated with relaxed per-run thresholds. Both ceilings can be made
stricter through the verifier CLI.

Export reviewed OCR feedback and enforce offline evaluation gates:

```bash
vehicle-dataset-export --export-id ocr-20260810-v1 --limit 500
python scripts/evaluate_ocr_dataset.py datasets/exports/ocr-20260810-v1 \
  --minimum-exact-accuracy 0.95 --minimum-character-accuracy 0.99 \
  --maximum-ece 0.08
```

The `/model-quality` console route reads the bounded `/api/model-quality` report.
See [Model Quality and Retraining Feedback](docs/MODEL_QUALITY_AND_RETRAINING.md)
for metric denominators, camera-grouped splits, checksums, retention pins, and
promotion limitations.

Before a production deployment, run the secret-safe static configuration/model
gate. The development defaults intentionally fail it:

```bash
python run_production_readiness.py \
  --config configs/default.yaml \
  --base-directory "$PWD" \
  --output output/production-readiness.json \
  --strict-warnings
```

See [Production Readiness Gate](docs/PRODUCTION_READINESS.md) for checked
invariants and the required live acceptance steps that remain outside a static
preflight. The hardened application profile, file-secret contract, OIDC/PKCE
requirements, and startup commands are in
[Production deployment](docs/PRODUCTION_DEPLOYMENT.md).

## Verification

The default test run is self-contained and uses an in-memory/JSONL persistence
path. Opt in to real local services with environment variables:

```bash
pytest
TEST_MONGODB_URI=mongodb://127.0.0.1:27017 pytest tests/integration/test_mongo_repository.py
TEST_REDIS_URL=redis://127.0.0.1:6379/15 pytest tests/integration/test_redis_streams.py
TEST_REDIS_URL=redis://127.0.0.1:6379/15 pytest tests/integration/test_realtime_redis.py
TEST_REDIS_URL=redis://127.0.0.1:6379/15 pytest tests/integration/test_live_monitor_redis.py
TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
  TEST_MONGODB_URI=mongodb://127.0.0.1:27017 \
  pytest tests/integration/test_redis_streams.py
TEST_MINIO_ENDPOINT=127.0.0.1:9000 pytest tests/integration/test_minio_storage.py

# Complete real-service acceptance suite:
TEST_MONGODB_URI=mongodb://127.0.0.1:27017 \
  TEST_REDIS_URL=redis://127.0.0.1:6379/15 \
  TEST_MINIO_ENDPOINT=127.0.0.1:9000 pytest -ra
```

CI additionally runs a required fail-closed real-service gate against the pinned
Compose MongoDB replica set, Redis, and MinIO services. To reproduce that exact
selection locally after starting the services, run:

```bash
TEST_MONGODB_URI='mongodb://127.0.0.1:27017/?replicaSet=rs0&directConnection=true' \
TEST_REDIS_URL='redis://127.0.0.1:6379/15' \
TEST_MINIO_ENDPOINT='127.0.0.1:9000' \
uv run --locked python scripts/run_real_service_tests.py
```

The command fails when any required connection setting is missing or any
selected service-backed integration test is skipped.
