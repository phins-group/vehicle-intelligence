# Roadmap

## Phase 1 — video MVP (accepted)

- Provider-agnostic vehicle/plate detectors, ByteTrack, PaddleOCR adapter.
- Quality gating, adaptive preprocessing, Vietnamese normalization, temporal
  voting, direction estimation, exactly-once finalization.
- Local/MinIO media storage, JSONL/Mongo event repositories, basic FastAPI query
  surface, unit/integration/pipeline tests.
- Acceptance record: [PHASE1_ACCEPTANCE.md](PHASE1_ACCEPTANCE.md).

## Phase 2 — camera production path (accepted)

- Implemented foundation: single-camera RTSP CLI, bounded latest-frame queue,
  capped reconnect, stream epochs, graceful shutdown, and in-memory source health.
- Acceptance record: [PHASE2_RTSP_ACCEPTANCE.md](PHASE2_RTSP_ACCEPTANCE.md).
- Implemented event path: versioned event codec, direct/Redis publisher adapters,
  consumer groups, stale-message reclaim, DLQ, ACK/delete lifecycle, idempotent
  MongoDB event worker, CLI, and Compose service.
- Acceptance record:
  [PHASE2_EVENT_BUS_ACCEPTANCE.md](PHASE2_EVENT_BUS_ACCEPTANCE.md).
- Implemented camera management: credential-safe CRUD/connection tests,
  AES-256-GCM MongoDB persistence, optimistic revisions, persisted latest health,
  and a host-native one-process-per-camera supervisor with crash isolation.
- Acceptance record:
  [PHASE2_CAMERA_MANAGEMENT_ACCEPTANCE.md](PHASE2_CAMERA_MANAGEMENT_ACCEPTANCE.md).
- Implemented policy path: normalized/temporal watchlists, allowlisted declarative
  rules, bounded priority evaluation, durable idempotent action claims,
  alert/log/guarded-HTTP handlers, alert lifecycle/cursor API, and Redis ACK after
  policy completion.
- Acceptance record:
  [PHASE2_POLICY_ENGINE_ACCEPTANCE.md](PHASE2_POLICY_ENGINE_ACCEPTANCE.md).
- Implemented security foundation: pluggable Bearer authenticator, SHA-256 API-key
  verifier, explicit `VIEWER`/`OPERATOR`/`ADMIN` permission matrix, authenticated
  alert actors, request correlation, secret-redacted append-only audit records,
  MongoDB indexes, and ADMIN audit query API.
- Acceptance record:
  [PHASE2_SECURITY_ACCEPTANCE.md](PHASE2_SECURITY_ACCEPTANCE.md).
- Implemented realtime path: post-policy Redis Pub/Sub notification, reconnecting
  API subscriber, bounded per-client queues and local replay, explicit gap
  recovery, authorized SSE/WebSocket endpoints, and realtime health.
- Acceptance record:
  [PHASE2_REALTIME_ACCEPTANCE.md](PHASE2_REALTIME_ACCEPTANCE.md).
- Implemented operator-console foundation: Angular 21 standalone/lazy routes,
  API-key session handling, role-aware controls, dashboard, cursor/live event
  explorer, camera operations, alert workflow, system health, WebSocket gap
  recovery, and same-origin Nginx packaging.
- Acceptance record:
  [PHASE2_OPERATOR_DASHBOARD_ACCEPTANCE.md](PHASE2_OPERATOR_DASHBOARD_ACCEPTANCE.md).
- Implemented policy console: typed watchlist CRUD with validity lifecycle,
  structured allowlisted rule authoring, optimistic revision conflicts, stable
  action IDs, ADMIN-only mutation affordances, audited destructive confirmations,
  responsive lazy routes, and policy utility tests.
- Acceptance record:
  [PHASE2_POLICY_UI_ACCEPTANCE.md](PHASE2_POLICY_UI_ACCEPTANCE.md).
- Implemented plate investigation: normalized exact search with opaque cursor,
  shareable Angular query route, loaded-only evidence summaries, chronological
  camera timeline, event drawer/handoff, bounded browser state, and explicit
  separation between plate observations and global vehicle identity.
- Acceptance record:
  [PHASE2_VEHICLE_SEARCH_UI_ACCEPTANCE.md](PHASE2_VEHICLE_SEARCH_UI_ACCEPTANCE.md).
- Implemented signed evidence presentation: event-scoped RBAC lookup, safe-key
  validation, MinIO existence checks, bounded presigned URLs, explicit missing
  objects, no-store responses, automatic Angular expiry refresh, and shared
  responsive media rendering in event/investigation drawers.
- Acceptance record:
  [PHASE2_SIGNED_MEDIA_ACCEPTANCE.md](PHASE2_SIGNED_MEDIA_ACCEPTANCE.md).
- Implemented human OCR review: immutable AI prediction, schema-v2 final plate,
  optimistic review revisions, OPERATOR/ADMIN RBAC, correction/confirmation
  audit, indexed final-plate search, idempotent dataset samples, and Angular
  review queue using signed evidence.
- Acceptance record:
  [PHASE2_HUMAN_OCR_REVIEW_ACCEPTANCE.md](PHASE2_HUMAN_OCR_REVIEW_ACCEPTANCE.md).
- Implemented Live Monitor: model-agnostic overlay contracts, latest-only edge
  reporter, background bounded JPEG, isolated Redis Pub/Sub, bounded exact-frame
  API, health, RBAC, and configurable Angular SVG overlays.
- Acceptance record:
  [PHASE2_LIVE_MONITOR_ACCEPTANCE.md](PHASE2_LIVE_MONITOR_ACCEPTANCE.md).
- Implemented ONVIF/multi-camera ingress hardening: bounded credential-free
  WS-Discovery, explicit batch admission outcomes, configured/active camera
  capacities, start-rate limiting, per-camera capped crash backoff, RBAC/audit,
  and Angular discovery workflow.
- Acceptance record:
  [PHASE2_ONVIF_MULTI_CAMERA_ACCEPTANCE.md](PHASE2_ONVIF_MULTI_CAMERA_ACCEPTANCE.md).
- Implemented observability/retention: low-cardinality Prometheus endpoints for
  API/camera and worker metrics, optional OTLP/HTTP FastAPI traces with log
  correlation, enriched persisted camera telemetry, leased bounded
  MongoDB/MinIO cleanup, dataset pins, and conservative lifecycle reconciliation.
- Acceptance record:
  [PHASE2_OBSERVABILITY_RETENTION_ACCEPTANCE.md](PHASE2_OBSERVABILITY_RETENTION_ACCEPTANCE.md).
- Implemented production security hardening: online AES-GCM keyring rotation,
  OIDC/JWKS centralized identity, shared-client replica-set transactions for
  resource/audit atomicity, server-owned Bearer/HMAC external-action credentials,
  and retry-aware circuit breakers.
- Acceptance record:
  [PHASE2_PRODUCTION_SECURITY_ACCEPTANCE.md](PHASE2_PRODUCTION_SECURITY_ACCEPTANCE.md).
- Implemented production validation: bounded Mongo outage detection and pending
  reclaim, partial-action retry without repeated completed side effects, Redis
  realtime/live reconnect, poison-message isolation, and thresholded burst/soak
  benchmark tooling.
- Acceptance record:
  [PHASE2_PRODUCTION_VALIDATION_ACCEPTANCE.md](PHASE2_PRODUCTION_VALIDATION_ACCEPTANCE.md).

### Remaining milestones: 0

## Phase 3 — multi-camera intelligence (accepted)

1. **Accepted:** vehicle identity/fingerprint and versioned embedding/vector
   ports. Each event receives a safe bootstrap identity; equal plates are not
   silently merged, vectors are separately versioned, and similarity is limited
   to pre-filtered candidates. Acceptance record:
   [PHASE3_IDENTITY_FOUNDATION_ACCEPTANCE.md](PHASE3_IDENTITY_FOUNDATION_ACCEPTANCE.md).
2. **Accepted:** directed camera topology and travel-time constrained candidate
   generation with audited CRUD, indexed inbound windows, and hard query caps.
   Acceptance record:
   [PHASE3_TOPOLOGY_CANDIDATES_ACCEPTANCE.md](PHASE3_TOPOLOGY_CANDIDATES_ACCEPTANCE.md).
3. **Accepted:** versioned multi-signal ReID scoring plus transactional human
   merge/split review, optimistic revisions, idempotency, and audit. Acceptance:
   [PHASE3_REID_REVIEW_ACCEPTANCE.md](PHASE3_REID_REVIEW_ACCEPTANCE.md).
4. **Accepted:** bounded event-derived journey generation, logical vehicle
   detail, directed-topology travel analysis, and cross-camera timeline UI.
   Acceptance record:
   [PHASE3_JOURNEY_UI_ACCEPTANCE.md](PHASE3_JOURNEY_UI_ACCEPTANCE.md).

## Phase 4 — optimization (three milestones)

1. **Accepted:** model/component benchmark contracts, real ONNX export/runtime,
   guarded TensorRT execution-provider path, artifact integrity, and regression
   gates. Acceptance record:
   [PHASE4_MODEL_OPTIMIZATION_ACCEPTANCE.md](PHASE4_MODEL_OPTIMIZATION_ACCEPTANCE.md).
2. **Accepted:** fair bounded multi-camera scheduling, real batch coordination,
   immutable edge manifests, non-root packaging, and deployment capacity gates.
   Acceptance record:
   [PHASE4_EDGE_SCHEDULER_ACCEPTANCE.md](PHASE4_EDGE_SCHEDULER_ACCEPTANCE.md).
3. **Accepted:** server-aggregated model-quality reporting/dashboard, leased and
   immutable camera-grouped OCR feedback exports, checksum verification,
   calibration/accuracy release gates, and final platform validation. Acceptance:
   [PHASE4_FINAL_ACCEPTANCE.md](PHASE4_FINAL_ACCEPTANCE.md).

## Post-acceptance operations

- Implemented a static, secret-safe production-readiness gate covering deployment
  posture and model artifact integrity. The development defaults intentionally
  fail closed; see [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md).
- Current roadmap milestones remaining: **0**. Site/model onboarding and live
  rollout validation are deployment inputs, not incomplete software milestones.
