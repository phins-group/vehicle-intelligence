# On-Prem Shadow Pilot Rollout Runbook (2–4 Cameras)

## Scope and operating invariant

This runbook takes one immutable release from lab evidence to an on-premises,
shadow-only pilot on two cameras, then optionally three or four cameras. The
pilot observes, stores, and presents results to authorized operators. It does
not control a physical device.

**Automatic barrier opening and every other fully autonomous physical action are
prohibited throughout this runbook.** Keep
`rule_engine.external_actions_enabled=false`, deny unapproved egress at the
network boundary, and verify the effective setting in the readiness evidence.
Passing this pilot is necessary but does not itself authorize a barrier. Any
later physical-action enablement requires a separate, signed safety/vendor
acceptance, change request, rollback drill, and accountable human owner.

The development defaults are intentionally not a deployable pilot
configuration. Use an independently managed, immutable pilot configuration and
secret-manager injection. Do not weaken a failed gate to obtain a pass.

Primary repository contracts:

- [Production readiness](PRODUCTION_READINESS.md)
- [Production failure, load, and soak validation](PRODUCTION_VALIDATION.md)
- [Model quality and retraining](MODEL_QUALITY_AND_RETRAINING.md)
- [Detector training](DETECTOR_TRAINING.md)
- [Model runtime optimization](MODEL_OPTIMIZATION.md)
- [Fair scheduling and edge deployment](EDGE_DEPLOYMENT.md)
- [Security and audit](SECURITY_AND_AUDIT.md)
- [Observability and retention](OBSERVABILITY_AND_RETENTION.md)
- [Dataset governance](PHINS_DATASET_GOVERNANCE.md)

## Owners, evidence, and acceptance values

Name one person for each role before work starts: release owner, site/operator
owner, ML-quality owner, infrastructure owner, security/privacy owner, legal
owner, and final go/no-go approver. One person may hold multiple roles, but no
technical owner may silently approve an unresolved legal or privacy gate.

Freeze these inputs in `decision/acceptance-contract.md` before measuring:

- release commit and image digests;
- configuration version and SHA-256, with secrets excluded;
- model names, versions, SHA-256 values, class maps, and edge-manifest SHA-256;
- the exact two initial camera IDs and the optional third/fourth camera IDs;
- business-approved accuracy, false-accept, unknown/review, latency, throughput,
  fairness, drop, resource-growth, recovery, RPO, RTO, and soak-duration limits;
- owners, evidence locations, maintenance window, and rollback version.

Do not copy development-host measurements into this contract. Detector release
thresholds come from the reviewed `configs/model-training.yaml`; OCR and runtime
thresholds are explicit CLI inputs; event-path thresholds are deployment SLO
inputs. Record the chosen values and their approvers rather than relying on a
tool's defaults. Fill every value below from that signed contract before running
any phase. Angle-bracket values are deliberately non-runnable placeholders.
Create the mode-0700 evidence directory on an approved encrypted on-premises
volume; it must not contain credentials, RTSP URLs, bearer tokens, or signed
media URLs. Run from the repository root with `uv` configured for the project's
pinned Python and lockfile; repository Python commands deliberately use
`uv run --locked python` rather than assuming a global `python` executable.

```bash
export PILOT_ID='<site>-<release>'
export PILOT_CONFIG='/approved/config/pilot.yaml'
export EVIDENCE_ROOT="/approved/evidence/$PILOT_ID"
export PERFORMANCE_GATE_DIR="$EVIDENCE_ROOT/decision/performance-gates"

export CAMERA_ID='<approved-camera-id>'
export VEHICLE_DATASET='/approved/datasets/vehicle-gold'
export PLATE_DATASET='/approved/datasets/plate-gold'
export OCR_GOLD_EXPORT='/approved/datasets/ocr-gold-export'
export VEHICLE_MODEL='/approved/models/vehicle.onnx'
export PLATE_MODEL='/approved/models/vietnam-plate.onnx'
export VEHICLE_MODEL_NAME='<approved-vehicle-model-name>'
export PLATE_MODEL_NAME='<approved-plate-model-name>'
export VEHICLE_MODEL_VERSION='<approved-vehicle-model-version>'
export PLATE_MODEL_VERSION='<approved-plate-model-version>'
export VEHICLE_PROVIDER='<approved-vehicle-provider>'
export PLATE_PROVIDER='<approved-plate-provider>'
export VEHICLE_MODEL_SHA256='<approved-64-hex-sha256>'
export PLATE_MODEL_SHA256='<approved-64-hex-sha256>'
export VEHICLE_DEVICE='<approved-runtime-device>'
export PLATE_DEVICE='<approved-runtime-device>'
export VEHICLE_EXECUTION_PROVIDER='<approved-execution-provider>'
export PLATE_EXECUTION_PROVIDER='<approved-execution-provider>'

export REPRESENTATIVE_VEHICLE_IMAGE='/approved/holdout/vehicle-gate.jpg'
export REPRESENTATIVE_PLATE_IMAGE='/approved/holdout/plate-gate.jpg'
export REPRESENTATIVE_VIDEO='/approved/holdout/camera-window.mp4'
export BENCHMARK_FRAMES='<approved-positive-frame-count>'

export PILOT_SOAK_SECONDS='<approved-seconds>'
export PILOT_EVENT_RATE='<approved-events-per-second>'
export MIN_EVENT_THROUGHPUT='<approved-events-per-second>'
export MAX_EVENT_P95_MS='<approved-milliseconds>'
export MAX_EVENT_ERROR_RATE='<approved-ratio>'
export MAX_RSS_GROWTH_MB='<approved-megabytes>'
export MIN_FAIRNESS='<approved-ratio>'
export MAX_DROP_RATIO='<approved-ratio>'
export MAX_EDGE_P95_MS='<approved-milliseconds>'
export MIN_OCR_EXACT='<approved-ratio>'
export MIN_OCR_CHARACTER='<approved-ratio>'
export MAX_OCR_ECE='<approved-ratio>'
export MIN_VEHICLE_FPS='<approved-frames-per-second>'
export MAX_VEHICLE_P95_MS='<approved-milliseconds>'
export MIN_PLATE_FPS='<approved-frames-per-second>'
export MAX_PLATE_P95_MS='<approved-milliseconds>'
export PILOT_CAMERA_FPS='<approved-frames-per-second-per-camera>'
export CAPACITY_DURATION_SECONDS='<approved-seconds>'
export PILOT_BATCH_SIZE='<approved-positive-batch-size>'
export PILOT_MAX_FRAME_AGE_MS='<approved-milliseconds>'

export DEPENDENCY='<mongodb|redis|minio>'

install -d -m 0700 \
  "$EVIDENCE_ROOT"/{preflight,models,rtsp,capacity,failures,recovery,decision} \
  "$PERFORMANCE_GATE_DIR"
```

Every phase produces a dated operator transcript plus immutable JSON reports.
The release owner records command exit status and hashes each final evidence
artifact. A screenshot alone is not gate evidence.

## Phase 0 — Freeze the shadow boundary

### Entry criteria

- Pilot cameras, network zone, edge node, and accountable owners are known.
- A candidate release, rollback release, and maintenance window exist.
- The acceptance contract has no blank owner or threshold.

### Execute

1. Put cameras, edge, API, MongoDB, Redis, and MinIO on the approved on-premises
   network paths. Restrict management ports to administrative sources.
2. Keep external actions, remote training, restricted dataset sync, and
   unapproved telemetry export disabled. Confirm the firewall independently of
   application configuration.
3. Configure human-readable camera IDs, time synchronization, ROI/crossing line,
   direction, sampling rate, and the expected day/night schedule. An ADMIN must
   create cameras explicitly; discovery never provisions RTSP credentials.
4. Pin the release, models, configuration, edge manifest, and container images.
   Mutable tags, implicit model downloads, and unversioned configuration are
   blockers.
5. Confirm the durable finalization outbox is enabled and its
   `storage.output_directory` is a persistent, capacity-monitored volume. Size
   `finalization_outbox.maximum_entries`, `maximum_bytes`, and
   `maximum_entry_bytes` from the approved outage/camera envelope. It is a hard
   capacity gate, not a drop-oldest buffer.

### Exit criteria

- The shadow-only boundary is signed by the release, site, and safety owners.
- Firewall and effective configuration both prevent physical/external actions.
- All immutable inputs and the rollback target are hash-recorded.
- The pilot cannot start if the outbox is disabled or stored on ephemeral disk.

## Phase 1 — Legal, data-governance, privacy, and secrets gates

### Entry criteria

- Phase 0 passed.
- Model and dataset provenance manifests are available to legal and ML owners.

### Execute

#### License decision

Build one inventory covering source code, runtime packages, every vehicle/plate/
OCR/ReID artifact, training data, evaluation data, and annotation-assistant
models. The repository's source-code license does not relicense a model or
dataset.

The legal owner must sign exactly one applicable Ultralytics decision for the
intended commercial, on-premises use and distribution model:

1. documented commercial/enterprise terms cover the exact runtime and model
   path; or
2. legal has reviewed and accepted all obligations of the applicable
   open-source path and documented how the deployment complies.

An unresolved decision is **NO-GO**. The generic community plate checkpoint in
the repository documentation is wiring/smoke evidence only and must not become
the pilot checkpoint. Dataset entries with unknown or review-required source
rights are not acceptance evidence. Record artifact hashes, source revisions,
license texts/links, decision owner, scope, and expiry/renewal conditions.

#### Privacy and retention

The privacy owner documents purpose, lawful/approved collection basis, notice or
signage, access roles, incident process, subject-request process where
applicable, and the smallest camera view/ROI needed for the purpose. Treat plate
images as restricted vehicle-identifier data. Do not upload continuous site
video or the gold holdout to an external service during this on-premises pilot.

Set retention values from the approved policy under `retention`. Canonical event
retention must be at least every canonical media window. Enable and monitor the
retention worker; align backup expiry with the policy; keep debug artifacts
disabled. Confirm MinIO lifecycle rules do not bypass dataset pins or canonical
media coordination. Signed-media URLs remain short-lived and must not enter
access logs, tickets, chat, or evidence files.

#### Secrets and identity

Inject MongoDB, Redis, MinIO, RTSP, OIDC/API, camera-encryption, and any allowed
integration credentials from the approved secret manager. Do not place raw
values in YAML, manifests, command arguments, shell history, evidence, or source
control. Use TLS, production authentication, at least one active ADMIN, least
privilege, and separate pilot/service/operator identities.

Rotate any credential that has left the approved secret boundary before pilot
admission. Rehearse API-key overlap (add new, move clients, remove old) or the
equivalent OIDC procedure. Rehearse camera encryption-key rotation first with:

```bash
uv run --locked vehicle-credential-rotation --config "$PILOT_CONFIG" --dry-run
```

Run the real rotation only during the approved window. Remove an old camera
decrypt key only after every camera document reports the active key. Database,
object-store, broker, RTSP, and identity-provider credentials follow their
site-owned rotation runbooks; attach redacted success evidence, never values.

### Exit criteria

- Legal signed the Ultralytics/model/data inventory decision with no unresolved
  production artifact.
- Privacy approved camera views, on-premises data flow, access, retention,
  backup expiry, and deletion/incident procedures.
- Authentication, TLS, RBAC, secret-manager delivery, and rotation/rollback were
  tested without exposing a secret.
- External actions and unapproved data egress remain disabled.

## Phase 2 — Static preflight, real services, and recovery baseline

### Entry criteria

- Phases 0–1 passed.
- Dedicated test namespaces and an isolated restore target exist.

### Execute

Validate Compose rendering and the production configuration. The readiness
report is intentionally secret-safe; retain its JSON output.

```bash
docker compose config --quiet
docker compose -f docker-compose.edge.yml --profile edge config --quiet

uv run --locked python run_production_readiness.py \
  --config "$PILOT_CONFIG" \
  --base-directory . \
  --output "$EVIDENCE_ROOT/preflight/production-readiness.json" \
  --strict-warnings
```

Run the required real MongoDB-replica-set, Redis, and MinIO selection against
dedicated test instances. Inject `TEST_MONGODB_URI`, `TEST_REDIS_URL`, and
`TEST_MINIO_ENDPOINT` through the test secret boundary before invoking the
runner; do not put their values in the transcript.

```bash
uv run --locked python scripts/run_real_service_tests.py \
  >"$EVIDENCE_ROOT/preflight/real-services.log" 2>&1
```

Start the pilot control plane only with production overrides in effect, then
check process and traffic-admission probes:

```bash
docker compose \
  --profile event-driven --profile observability --profile maintenance \
  up -d mongodb mongodb-init redis minio api web event-worker prometheus retention-worker

curl --fail --silent --show-error http://127.0.0.1:8000/livez \
  >"$EVIDENCE_ROOT/preflight/livez.json"
curl --fail --silent --show-error http://127.0.0.1:8000/readyz \
  >"$EVIDENCE_ROOT/preflight/readyz.json"
```

Ingress targets `/readyz`; container restart health targets `/livez`. A failed
canonical MongoDB ping must remove API traffic admission. MinIO, realtime, or
camera-management degradation must remain visible without falsely declaring the
process dead.

### Backup and isolated restore rehearsal

The repository does not bundle a production backup product. The infrastructure
owner must use the site-approved MongoDB and object-storage tools and attach the
exact versioned commands/run identifiers. The rehearsal must:

1. create known pilot sentinel events with all enabled media kinds and record
   event IDs, object keys, byte sizes, and SHA-256 values;
2. quiesce camera ingestion, allow the finalization outbox and Redis stream to
   drain, and record the consistent cutoff time;
3. back up MongoDB, the private MinIO bucket, immutable configuration/model/edge
   manifests, audit evidence, and every persistent edge `/data` outbox volume;
4. include encrypted camera credentials but never decrypt or print them in the
   backup transcript;
5. restore into a new isolated database, bucket, API endpoint, and edge volume;
   never use a destructive restore option against the pilot target;
6. verify event/document counts, sampled event IDs, audit records, object counts,
   every sentinel media checksum, one authorized signed-media read, and outbox
   replay after restart;
7. measure RPO/RTO against the signed acceptance contract, then destroy the
   isolated restore according to policy.

Redis is a durable delivery boundary but MongoDB and MinIO are canonical event
and media stores. Still preserve/recover broker state according to the site's
chosen RPO, and prove pending entries do not become silent loss or duplicate
canonical documents. A volume snapshot without a successful isolated restore is
not backup evidence.

### Exit criteria

- Static readiness and the fail-on-skip real-service gate exit zero.
- `/livez` and `/readyz` match their documented semantics.
- The isolated restore preserves sentinel event/media integrity and meets the
  approved RPO/RTO.
- Backup access, encryption, retention, and deletion are privacy-approved.

## Phase 3 — Gold holdout and model release gates

### Entry criteria

- Phase 2 passed.
- Licensed candidate artifacts and a separately governed gold holdout exist.

### Gold holdout contract

Freeze a representative, immutable, human-labeled holdout before evaluating the
candidate. It must be disjoint by site/camera/time and real passage or track,
never by random individual frames. It must not be used for training, tuning,
threshold selection, or annotation-assistant acceptance. Human-review feedback
is useful regression data but is selection-biased and cannot replace this set.

Record coverage and denominators for each approved operational slice, including
camera, day/night, readable/unreadable, one-line/two-line, vehicle class,
motorcycle, glare, blur, small plate, occlusion, and oblique angle. With only
four cameras, use operational time/track holdouts plus explicit unseen-camera
groups as described in `DETECTOR_TRAINING.md`.

### Execute detector gates

Verify immutable datasets, create predictions through the same provider boundary
used in production, and evaluate vehicle and plate roles. The evaluator enforces
the reviewed gates in `configs/model-training.yaml` and fails when required
slices or critical classes are missing.

```bash
uv run --locked python run_model_training.py --config configs/model-training.yaml \
  verify-dataset "$VEHICLE_DATASET"
uv run --locked python run_model_training.py --config configs/model-training.yaml \
  verify-dataset "$PLATE_DATASET"

uv run --locked python run_model_training.py --config configs/model-training.yaml \
  predict --role vehicle "$VEHICLE_DATASET" --split test \
  --runtime-config "$PILOT_CONFIG" --provider "$VEHICLE_PROVIDER" \
  --model "$VEHICLE_MODEL" --model-name "$VEHICLE_MODEL_NAME" \
  --model-version "$VEHICLE_MODEL_VERSION" --model-hash "$VEHICLE_MODEL_SHA256" \
  --device "$VEHICLE_DEVICE" \
  --output "$EVIDENCE_ROOT/models/vehicle-predictions.json"
uv run --locked python run_model_training.py --config configs/model-training.yaml \
  evaluate --role vehicle "$VEHICLE_DATASET" --split test \
  --predictions "$EVIDENCE_ROOT/models/vehicle-predictions.json" \
  --output "$EVIDENCE_ROOT/models/vehicle-evaluation.json"

uv run --locked python run_model_training.py --config configs/model-training.yaml \
  predict --role plate "$PLATE_DATASET" --split test \
  --runtime-config "$PILOT_CONFIG" --provider "$PLATE_PROVIDER" \
  --model "$PLATE_MODEL" --model-name "$PLATE_MODEL_NAME" \
  --model-version "$PLATE_MODEL_VERSION" --model-hash "$PLATE_MODEL_SHA256" \
  --device "$PLATE_DEVICE" \
  --output "$EVIDENCE_ROOT/models/plate-predictions.json"
uv run --locked python run_model_training.py --config configs/model-training.yaml \
  evaluate --role plate "$PLATE_DATASET" --split test \
  --predictions "$EVIDENCE_ROOT/models/plate-predictions.json" \
  --output "$EVIDENCE_ROOT/models/plate-evaluation.json"
```

If provider-specific prediction arguments are required, record them in the
acceptance contract and verify `run_model_training.py ... predict --help`; do not
change them after opening the gold results.

Evaluate OCR on the immutable labeled export with the signed thresholds:

```bash
uv run --locked python scripts/evaluate_ocr_dataset.py "$OCR_GOLD_EXPORT" \
  --minimum-exact-accuracy "$MIN_OCR_EXACT" \
  --minimum-character-accuracy "$MIN_OCR_CHARACTER" \
  --maximum-ece "$MAX_OCR_ECE" \
  --output "$EVIDENCE_ROOT/models/ocr-evaluation.json"
```

Measure both detector roles on a labeled representative image from the gold set
and on the exact pilot node/runtime. Marking an input representative is an
auditable assertion, not a substitute for provenance.

```bash
uv run --locked python scripts/benchmark_detector.py \
  --model "$VEHICLE_MODEL" --provider "$VEHICLE_PROVIDER" --role vehicle \
  --model-name "$VEHICLE_MODEL_NAME" --model-version "$VEHICLE_MODEL_VERSION" \
  --model-hash "$VEHICLE_MODEL_SHA256" --device "$VEHICLE_DEVICE" \
  --execution-provider "$VEHICLE_EXECUTION_PROVIDER" \
  --image "$REPRESENTATIVE_VEHICLE_IMAGE" --representative-input \
  --minimum-fps "$MIN_VEHICLE_FPS" --maximum-p95-ms "$MAX_VEHICLE_P95_MS" \
  --output "$EVIDENCE_ROOT/models/vehicle-runtime.json"

uv run --locked python scripts/benchmark_detector.py \
  --model "$PLATE_MODEL" --provider "$PLATE_PROVIDER" --role plate \
  --model-name "$PLATE_MODEL_NAME" --model-version "$PLATE_MODEL_VERSION" \
  --model-hash "$PLATE_MODEL_SHA256" --device "$PLATE_DEVICE" \
  --execution-provider "$PLATE_EXECUTION_PROVIDER" \
  --image "$REPRESENTATIVE_PLATE_IMAGE" --representative-input \
  --minimum-fps "$MIN_PLATE_FPS" --maximum-p95-ms "$MAX_PLATE_P95_MS" \
  --output "$EVIDENCE_ROOT/models/plate-runtime.json"
```

Record end-to-end exact-plate correctness, false accepts, unknowns, human
corrections, and latency with explicit denominators for every camera/slice. A
high aggregate score cannot waive a failed safety-critical or privacy-approved
slice.

### Exit criteria

- Vehicle, plate, and OCR evaluations exit zero with immutable checksum lineage.
- Every required camera/time/condition slice is present and passes its signed
  gate; no result was tuned after opening the gold set.
- Runtime provider availability and model hashes fail closed on the target node.
- The ML and legal owners sign the exact packaged candidates. Promotion remains
  manual.

## Phase 4 — One-camera representative RTSP full-pipeline qualification

### Entry criteria

- Phase 3 passed.
- The first camera is privacy-approved, time-synchronized, and configured by an
  ADMIN without exposing its RTSP URL.

### Execute

For each camera, collect an approved bounded natural clip covering its expected
geometry and conditions, then benchmark the real vehicle → tracking → plate →
OCR path. Store the clip only for the approved period.

```bash
uv run --locked python scripts/benchmark_pipeline.py "$REPRESENTATIVE_VIDEO" \
  --config "$PILOT_CONFIG" \
  --vehicle-model "$VEHICLE_MODEL" \
  --plate-model "$PLATE_MODEL" \
  --max-frames "$BENCHMARK_FRAMES" \
  >"$EVIDENCE_ROOT/rtsp/${CAMERA_ID}-pipeline.json"
```

Run the live source in shadow mode through the complete production path:

```text
RTSP -> decode/sample -> detectors/OCR/tracking -> durable finalization outbox
     -> MinIO media -> Redis Streams -> event worker -> MongoDB -> API/UI/review
```

Start the supervisor under the site's service manager, with the pilot config and
secrets injected outside the command line:

```bash
uv run --locked vehicle-camera-supervisor \
  --config "$PILOT_CONFIG" \
  --worker-config "$PILOT_CONFIG"
```

Observe one full approved window that includes normal passages, no-plate/
unreadable cases, lighting transition where applicable, a controlled RTSP
disconnect/reconnect, and graceful worker restart. Review a stratified sample
against human ground truth. Confirm:

- source/decode/inference FPS, queue depth, drops, reconnects, active tracks, and
  stage latency remain within the signed contract;
- reconnect changes stream epoch and does not duplicate one logical finalized
  track;
- every canonical event media reference resolves to the correct private object;
- event/model metadata contains the approved names, versions, and hashes;
- partial/low-confidence results enter human review instead of becoming a false
  confirmed plate;
- outbox entries drain after healthy delivery and survive a process restart;
- no RTSP credential, plate, event ID, signed URL, or exception text appears in
  metric labels or unsafe logs;
- no external/physical action is attempted.

### Exit criteria

- The single-camera full path passes every signed operational and accuracy gate.
- Reconnect/restart causes no unexplained event loss, wrong media association, or
  duplicate canonical MongoDB document.
- Queues and outbox return to the recorded healthy steady state after quiescing.
- Operator review and rollback procedures are usable without privileged data
  leakage.

## Phase 5 — Two-camera shadow soak and capacity gate

### Entry criteria

- Phase 4 passed independently for both initial cameras.
- Failure-recovery traps and an on-call operator are ready.

### Execute capacity evidence

Run the scheduler capacity benchmark on the exact pilot node/model with a
representative site image and the signed two-camera offered load:

```bash
uv run --locked python scripts/benchmark_edge_capacity.py \
  --config "$PILOT_CONFIG" \
  --model "$VEHICLE_MODEL" --provider "$VEHICLE_PROVIDER" \
  --model-version "$VEHICLE_MODEL_VERSION" \
  --device "$VEHICLE_DEVICE" \
  --execution-provider "$VEHICLE_EXECUTION_PROVIDER" \
  --image "$REPRESENTATIVE_VEHICLE_IMAGE" \
  --cameras 2 --camera-fps "$PILOT_CAMERA_FPS" \
  --duration-seconds "$CAPACITY_DURATION_SECONDS" \
  --batch-size "$PILOT_BATCH_SIZE" \
  --maximum-frame-age-ms "$PILOT_MAX_FRAME_AGE_MS" \
  --minimum-fairness "$MIN_FAIRNESS" \
  --maximum-drop-ratio "$MAX_DROP_RATIO" \
  --maximum-p95-latency-ms "$MAX_EDGE_P95_MS" \
  --output "$EVIDENCE_ROOT/capacity/edge-2-camera.json"
```

Run the real Redis → worker → Mongo path at the signed rate against dedicated
benchmark namespaces. Supply service locations through the approved test-secret
environment; the benchmark cleans its unique namespace.

```bash
uv run --locked python scripts/benchmark_event_path.py \
  --events 1 --soak-seconds "$PILOT_SOAK_SECONDS" \
  --rate "$PILOT_EVENT_RATE" \
  --minimum-throughput "$MIN_EVENT_THROUGHPUT" \
  --maximum-p95-batch-ms "$MAX_EVENT_P95_MS" \
  --maximum-error-rate "$MAX_EVENT_ERROR_RATE" \
  --maximum-rss-growth-mb "$MAX_RSS_GROWTH_MB" \
  >"$EVIDENCE_ROOT/capacity/event-path-soak.json"
```

Run both real RTSP cameras concurrently for the approved soak duration. Monitor
`/readyz`, `/metrics`, camera health, process/container resources, Redis pending
and DLQ counts, MinIO errors, MongoDB/action idempotency, retention failures, and
durable outbox capacity. Inspect beginning, middle, lighting-transition, and end
samples against ground truth. After stopping new ingestion, require every queue
to drain or have an explained, approved residual.

### Exit criteria

- Thresholded capacity and event-path reports exit zero.
- The real two-camera soak meets the signed duration and load without starvation,
  unbounded growth, silent drops, or wrong cross-camera associations.
- Redis pending/DLQ and each outbox return to the accepted quiescent state.
- Mongo event cardinality and MinIO media integrity reconcile with sampled source
  passages and no autonomous action occurred.

## Phase 6 — Failure drills and rollback rehearsal

### Entry criteria

- Phase 5 passed.
- The infrastructure owner confirmed that every drill is bounded to the pilot
  and that recovery access is available.

### Dependency drill harness

Exercise one dependency at a time. Install the recovery trap before pausing it,
validate the allowlist, and use the approved observation window:

```bash
set -euo pipefail
case "$DEPENDENCY" in mongodb|redis|minio) ;; *) exit 2 ;; esac

restore_dependency() {
  docker compose unpause "$DEPENDENCY" >/dev/null 2>&1 || true
}
trap 'restore_dependency' EXIT INT TERM
docker compose pause "$DEPENDENCY"
# Observe only for the bounded, pre-approved drill window.
docker compose unpause "$DEPENDENCY"
trap - EXIT INT TERM
```

Record detection time, visible state, event/media behavior, recovery time,
operator decision, and post-recovery reconciliation for every row:

| Drill | Required behavior and evidence |
| --- | --- |
| One RTSP source unavailable, then corrupt packets/resolution change | Only that camera becomes OFFLINE/degraded; retries are capped; other camera remains fair; reconnect/epoch and finalization are correct. |
| MongoDB unavailable | `/readyz` becomes `503`; Redis message remains recoverable; ambiguous retries do not create a second canonical event; readiness recovers after a bounded ping. |
| Redis unavailable/restarted | Existing REST history remains usable; camera finalization stays in the durable outbox; recovery replays delivery and downstream idempotency prevents duplicate Mongo/action state. |
| MinIO unavailable | `/readyz` reports degradation without declaring the process dead; the complete event/media unit remains in the outbox; media is restored before event publication on replay. |
| Event worker killed after delivery ambiguity | Redis pending/stale claim recovers; unique event/action keys make redelivery idempotent; DLQ remains bounded. |
| Edge process/node restart after staging | Persistent `/data` survives; outbox integrity is verified and healthy entries replay; no completed entry is silently discarded. |
| Outbox hard capacity | In an isolated copy with deliberately small limits, admission fails closed and never deletes the oldest entry. Do not fill the live pilot disk. |
| Outbox accidental corruption or uninformed replacement | Test only an isolated copied entry by truncating or changing bytes without recomputing its checksums; replay stops/fails closed with a secret-safe error. The SHA-256 manifest is not an authenticity proof against a process that can rewrite the entry and recompute every checksum. Treat same-UID write access as a compromised node and **NO-GO**. Never edit live evidence to perform this drill. |
| Model hash/provider mismatch | Startup fails closed before inference; no CPU fallback is reported as the requested accelerator. |
| API restart and slow realtime client | Camera/event durability continues; clients recover with explicit gap behavior; process liveness and traffic readiness remain semantically correct. |

### Application/model rollback

Rollback is triggered by any gate breach, unexplained event/media loss, false
physical action, secret/privacy exposure, sustained resource growth, integrity
failure, inability to drain delivery state, or an operator invoking stop. Do not
restore a database backup merely to roll back application code or a model.

1. Disable/pause camera admission through the authenticated control plane and
   stop edge/supervisor workers first. Keep MongoDB, Redis, MinIO, API, and the
   event worker available for reconciliation unless the incident requires
   isolation.
2. Preserve logs, readiness JSON, pending/DLQ counts, outbox state, and hashes.
   Do not delete or hand-edit a pending/corrupt entry.
3. Reapply the previous immutable image, configuration, model artifacts, and
   edge manifest from the Phase 0 rollback record. Verify every hash.
4. Run strict static readiness, `/livez`, `/readyz`, and one canary camera before
   re-admitting the second camera.
5. Drain/reconcile outbox and broker delivery, verify Mongo/MinIO cardinality and
   media checksums, then record the incident and decision. External actions stay
   disabled.

For a Compose-managed edge, stop ingestion without deleting its persistent
volume:

```bash
docker compose -f docker-compose.edge.yml --profile edge stop vision-edge
```

Never use `down -v`, delete the edge output volume, or remove broker/object-store
state during application rollback. Disaster restore uses the isolated, rehearsed
Phase 2 recovery procedure and requires the incident commander to name the exact
target and approved recovery point.

### Exit criteria

- Every required drill meets its signed detection/recovery/data-integrity gate.
- Rollback to the prior immutable release succeeds without deleting durable
  state or exposing secrets.
- Post-recovery queues, outbox, canonical records, media, and readiness reconcile.
- All drill changes are reverted and the shadow boundary remains enforced.

## Phase 7 — Expand to three/four cameras, then decide

### Entry criteria

- Phases 0–6 passed for two cameras with no open critical finding.
- ML, site, infrastructure, privacy, and release owners approve expansion.

### Execute

Add only one camera at a time. Repeat Phase 4 for the new camera before sharing
the node, then rerun the Phase 5 capacity command with `--cameras 3` and later
`--cameras 4`, using each actual offered FPS and the same signed thresholds.
Run the full multi-camera RTSP soak after each addition. Do not infer capacity
from the earlier development-host acceptance record.

Put only the final immutable detector-runtime and edge-capacity reports in a
dedicated verifier directory, then verify that set with the approved edge
ceilings:

```bash
# Copy the hash-verified final reports here without renaming their contents.
uv run --locked python scripts/verify_performance_gates.py "$PERFORMANCE_GATE_DIR" \
  --maximum-edge-drop-ratio "$MAX_DROP_RATIO" \
  --maximum-edge-p95-ms "$MAX_EDGE_P95_MS" \
  --output "$EVIDENCE_ROOT/decision/final-performance-gates.json"
```

Do not mix superseded or relaxed reports with release evidence.

### Exit criteria

- Each added camera passed its own accuracy, RTSP, reconnect, privacy, and
  operator-review gates.
- The exact final 2–4 camera topology passed full-pipeline soak, capacity,
  fairness, recovery, and backup/restore acceptance on the target hardware.
- No threshold changed after a failing measurement without creating a new,
  independently approved acceptance-contract revision and rerunning all affected
  phases.

## Final go/no-go checklist

Every box needs an owner, evidence path/hash, date, and decision. Any unchecked
blocking row is **NO-GO**.

- [ ] Immutable release, configuration, models, edge manifest, and rollback
  version are hash-recorded.
- [ ] Strict production readiness and required real-service CI pass without
  skips or ignored warnings.
- [ ] Ultralytics/runtime/model/dataset commercial-use decision is signed and
  every deployed artifact has resolved provenance.
- [ ] Gold holdout is site/camera/time/track-disjoint, immutable, representative,
  and untouched by training/tuning.
- [ ] Vehicle, plate, OCR, per-slice, runtime, and full-pipeline accuracy gates
  pass with explicit denominators.
- [ ] Exact target hardware and final camera count pass capacity, fairness,
  latency, drop, resource-growth, and soak gates.
- [ ] RTSP reconnect, dependency outage, process/node restart, idempotent replay,
  outbox capacity/integrity, and rollback drills pass.
- [ ] MongoDB/MinIO plus edge durable state were restored into an isolated target
  with verified event/media checksums and approved RPO/RTO.
- [ ] TLS, authentication, RBAC, audit, secret-manager delivery, credential
  rotation, network restrictions, and ingress controls pass.
- [ ] Privacy purpose, camera views, restricted-data handling, retention,
  backups, signed media, deletion, and incident response are approved.
- [ ] Operators can review uncertain plates, recognize degraded/offline states,
  stop ingestion, and invoke rollback.
- [ ] Pending/DLQ/outbox/canonical event/media state reconciles after quiescence;
  there is no unexplained loss, duplicate canonical state, or wrong media link.
- [ ] `rule_engine.external_actions_enabled=false` and the independent egress
  control prove that no barrier or autonomous physical action can execute.

The final decision is one of:

- **NO-GO / rollback** — any blocker or unexplained result remains;
- **CONTINUE SHADOW** — evidence is promising but more observation is required;
- **SHADOW PILOT ACCEPTED** — the 2–4 camera on-premises observation scope passed.

`SHADOW PILOT ACCEPTED` authorizes only the observation scope in this runbook.
It does not authorize automatic barrier opening, unattended enforcement, or any
other fully autonomous physical action.
