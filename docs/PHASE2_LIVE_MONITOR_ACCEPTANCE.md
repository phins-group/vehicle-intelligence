# Phase 2 Live Monitor Acceptance Record

## Decision

The bounded Live Monitor milestone passed engineering acceptance on 2026-08-09.
This accepts a model-agnostic overlay contract, latest-only edge reporting,
background JPEG encoding, an isolated Redis Pub/Sub channel, bounded API frame
rings, authenticated exact-sequence HTTP reads, health reporting, and a lazy
Angular monitor with configurable overlays.

It does not accept full-rate HLS/WebRTC delivery, continuous video recording,
durable preview history, a cross-replica shared frame cache, ONVIF discovery, or
production connection/load/soak limits.

## Accepted flow

```text
sampled inference frame
  -> vehicle/plate/track/ROI/line metadata
  -> one-slot latest-only edge reporter
  -> background capped resize/JPEG
  -> versioned size-limited Redis Pub/Sub packet
  -> reconnecting API subscriber
  -> bounded per-camera exact-frame ring
  -> READ_PLATFORM state request
  -> exact sequence JPEG request
  -> Angular source-coordinate SVG overlays
```

The canonical event Stream and event SSE/WebSocket remain image-free. Preview
loss cannot alter tracking, OCR voting, event finalization, persistence, rules,
or action processing.

## Automated evidence

- Ruff passed across the Python repository.
- The complete real-service suite passed **122 tests** against MongoDB 8, Redis
  8, and MinIO with no skips. New coverage includes strict codec validation,
  payload limits, bounded camera/frame eviction, staleness, reporter throttling
  and failure isolation, OpenCV resize/JPEG, API authorization, exact sequence
  reads, expired sequence handling, and real Redis Pub/Sub delivery.
- The pipeline test captured five preview observations while preserving the
  single canonical event path; its normalized plate and full-frame translated
  plate box were verified.
- Strict application/spec TypeScript typecheck passed. All **27 Vitest tests**
  passed, including deterministic overlay flags, sequence loading, SVG polygon
  conversion, and bounded labels.
- The Angular production build passed at **362.95 kB raw / 98.83 kB estimated
  initial transfer**. Live Monitor remains a lazy chunk at **13.34 kB raw /
  4.35 kB estimated transfer**.
- The production dependency audit reported zero vulnerabilities. The complete
  build-only graph reports seven moderate and four high transitive advisories
  through Angular tooling; Node/npm are absent from the Nginx runtime image.
- Compose validation and updated API/Nginx production image builds passed.
- A real 640x360 JPEG and overlay packet traversed host Redis, the containerized
  API subscriber, and same-origin Nginx. State became `LIVE`; plate text
  `51H-123.45`, frame ID, stream epoch, and sequence matched. The exact
  **16,281-byte** JPEG returned `200 image/jpeg`, `no-store`, and SHA-256
  `98a464603c84226e31cd19addde918c99d432c3d3ac7d5aae92cb06f7e00ef8d`.
- State JSON contained neither RTSP credentials nor JPEG/base64 content. A
  non-buffered sequence returned `410`; the shareable `/live-monitor?camera=...`
  SPA route returned `200` with CSP through Nginx.
- The scoped smoke camera was deleted through the API. MongoDB confirmed zero
  camera and camera-health records remained; its create/delete audit evidence
  remained append-only as designed.
- API and Nginx logs showed successful startup and expected `200`/`410`/`204`
  requests without an application traceback or credential exposure.

The one Python warning is the existing FastAPI/Starlette TestClient deprecation
for `httpx`. The host used unsupported odd Node 23 for one local build, which
disabled only an optional compiler cache; the verified container build uses the
pinned Node 24.12.0 and npm 11.10.1.

## Security and failure semantics

Every state, frame, and detailed health route requires `READ_PLATFORM`. Camera
existence and enabled state are checked before a frame is returned. State and
JPEG responses are `no-store`; the JPEG echoes sequence, frame, epoch, and
capture headers so the UI can reject a mismatched pair. RTSP URLs and encrypted
tokens never enter the live contract or browser.

Worker encoding uses one pending slot and a background thread; a newer frame
replaces stale work. Publish duration, payload size, frame rate, width, camera
count, and API frame depth are configured bounds. Encoder, contract, timeout,
Redis, subscriber, or malformed-message failures are isolated from inference and
canonical event creation.

The Angular client polls only while visible, fetches images as authenticated
Blobs, validates the echoed sequence, revokes superseded object URLs, and guards
against stale camera responses. It does not persist frames or credentials.

## Operational limits

- This is a low-rate operational preview, not a 25 FPS viewing service. JPEG
  base64 over Pub/Sub trades simplicity for overhead under explicit bounds.
- Redis Pub/Sub is best effort and has no replay. After loss or restart, the UI
  waits for a fresh frame.
- Frame sequences and rings are API-process-local. Multi-replica deployments
  require sticky routing or a shared bounded frame cache for state/frame pairs.
- No preview media is retained in MongoDB/MinIO; optional event clips remain a
  separate asynchronous feature.
- Local Compose smoke used development authentication and default local-service
  credentials. Production still requires TLS, API authentication/OIDC evolution,
  MongoDB/Redis/MinIO authentication and network isolation, and ingress rate/
  connection limits.
- Browser visual automation was not requested for this milestone. Compiler,
  deterministic utility tests, production builds, container routes, security
  headers, and protocol-level end-to-end behavior were verified; manual
  cross-browser visual and accessibility QA remains required before an external
  production release.
