# Phase 2 Signed Media Acceptance Record

## Decision

The event-scoped signed-media milestone passed engineering acceptance on
2026-08-09. This accepts RBAC-protected event media lookup, safe persisted-key
validation, real MinIO existence checks, separate internal/public endpoints,
offline short-lived signing, explicit missing-object state, private/no-store API
responses, and reusable Angular evidence rendering with expiry refresh.

It does not accept public bucket access, arbitrary-key signing, upload APIs,
local-filesystem HTTP serving, immediate URL revocation, per-read MongoDB audit,
MinIO lifecycle automation, or browser connection/load/soak limits.

## Accepted flow

```text
Angular event / plate-investigation drawer
  -> GET /api/events/{eventId}/media with Bearer identity
  -> FastAPI READ_PLATFORM authorization
  -> canonical VehicleEvent lookup in MongoDB
  -> validate persisted relative media keys
  -> MinIO stat through internal endpoint
  -> fixed-TTL signature for browser-reachable endpoint
  -> AVAILABLE or MISSING evidence slot
  -> image/video presentation and automatic pre-expiry refresh
```

There is no request field or route that signs a caller-provided object key.
Object bytes remain in MinIO and signed URLs remain transient browser state.

## Automated evidence

- The complete real MongoDB 8, Redis 8, and MinIO suite passed 106 Python tests.
  New coverage verifies mixed available/missing assets, safe-key rejection before
  any signature is issued, unknown events, unavailable local storage, no-store
  API headers, bounded configuration, RBAC, real signed downloads, and offline
  signing for a public hostname unreachable from the API container. Ruff and
  Python bytecode compilation also passed.
- Strict application/spec TypeScript typecheck passed. Twenty-two deterministic
  Vitest tests passed, including canonical media-slot ordering, missing evidence,
  pre-expiry refresh timing, expired URL retry, and invalid expiry handling.
- The Angular production build passed at 355.65 kB raw / 97.52 kB estimated
  initial transfer. The reusable media viewer is extracted into a shared lazy
  chunk at 7.13 kB raw / 2.52 kB estimated transfer.
- Production dependency audit reported zero vulnerabilities. The complete
  build-only graph reports seven moderate and four high transitive advisories
  through Angular tooling; Node/npm are absent from the Nginx runtime image.
- Compose validation and both updated API/Nginx images built successfully. The
  API image now installs the MinIO adapter explicitly; API, web, MongoDB, Redis,
  and MinIO became healthy/running and system health reported media access
  configured.
- A scoped event and a real 14,306-byte JPEG proved the same-origin Nginx API and
  signed-object gateway flow. The endpoint returned snapshot `AVAILABLE`, absent
  plate crop `MISSING`, a five-minute URL on browser host `localhost:4200`, and
  `Cache-Control: no-store, private`. Download returned `200 image/jpeg`; a
  modified signature returned `403`; an unknown event returned `404`.
- Adding an arbitrary `key=private/secret.jpg` query did not influence signing:
  the response retained the event's persisted snapshot key. Internal object
  lookup and public URL signing used separate clients; a configured region
  prevented the public signer from making a network call.
- The Nginx signed bucket-path proxy preserved the SigV4 Host, returned the JPEG
  under the SPA's own origin, disabled signed-query access logs, rejected
  modified signatures, and retained `img-src`/`media-src 'self'` CSP instead of
  allowing arbitrary remote media.
- With authentication temporarily enabled, missing and invalid credentials
  returned `401`; VIEWER, OPERATOR, and ADMIN each obtained event media with
  `200`. Compose was restored to development `auth=disabled`.
- The exact smoke event and exact MinIO object were deleted. MongoDB and MinIO
  confirmed zero scoped resources remained, and the cleaned event endpoint
  returned `404`.

One unchanged warning comes from FastAPI/Starlette's current TestClient use of
httpx. The unsupported host Node 23 runtime disabled an optional Angular compiler
cache but did not affect output; the verified container build uses pinned Node
24.12.0 and npm 11.10.1.

## Security and lifecycle semantics

The event ID is the only client selector. The service loads the canonical event
before reading media references and validates every key before concurrently
checking/signing any object. One unsafe reference fails the request without
issuing URLs for otherwise safe siblings. Provider errors return a generic `503`
without exposing storage credentials or internal endpoint details.

The internal MinIO client performs `stat_object`; the public client signs with a
configured region and therefore does not need to connect to the browser-facing
hostname. A missing object returns its durable key, null URL, content type, and
`MISSING` status so retention gaps remain investigable.

Presigned URLs are temporary bearer capabilities. Lifetime is server-controlled,
defaults to 300 seconds, and is constrained to 30–3,600 seconds. The UI opens
images with no referrer, preloads only clip metadata, refreshes thirty seconds
before expiry, rejects stale event responses, and destroys timers/URL state with
the drawer. It does not persist URLs in local or session storage.

## Operational limits

- `VIP_MINIO__PUBLIC_ENDPOINT` must be the operator gateway host, omit the URL
  scheme/path, and preserve that Host while proxying the signed bucket path;
  `PUBLIC_SECURE` selects HTTP versus HTTPS. Production requires TLS even though
  local Compose defaults to HTTP.
- API-key revocation cannot revoke an already issued URL. Urgent revocation must
  remove/rotate the object or use storage/edge policy; normal exposure is bounded
  by the short TTL.
- LocalMediaStorage remains a Phase 1 write adapter and deliberately returns
  media access unavailable through FastAPI.
- Read issuance is not written to MongoDB audit logs. Ingress and object-store
  access logs should record safe request metadata without full signed queries.
- Retention lifecycle and orphan-reference reconciliation remain future platform
  work.
- The in-app browser runtime was not used because browser visual testing was not
  requested. Compiler checks, deterministic tests, production builds, container
  routes, real object downloads, signature failure, RBAC, and cleanup were
  verified; manual cross-browser visual/keyboard QA remains required before an
  external production release.
