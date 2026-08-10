# Phase 2 Operator Dashboard Acceptance Record

## Decision

The Angular operator-console foundation passed engineering acceptance on
2026-08-09. This accepts the authenticated shell, role-aware operator controls,
dashboard, cursor/live vehicle-event exploration, camera create/test/lifecycle
actions, alert transitions, system-health views, WebSocket reconnect/gap recovery,
and same-origin Nginx packaging.

It does not accept watchlist/rule editors, vehicle identity/timeline, signed
media presentation, live video overlays, OIDC user lifecycle, or browser
connection load/soak limits.

## Accepted flow

    Angular feature route
      -> typed relative REST request
      -> same-origin Nginx
      -> FastAPI RBAC
      -> MongoDB-backed response

    Redis Pub/Sub event
      -> API bounded realtime hub
      -> Nginx WebSocket upgrade
      -> Angular RxJS stream
      -> stable-ID deduplication
      -> matching view update
      -> REST recovery after explicit gap

No frontend component reads MongoDB, Redis, MinIO, or an RTSP source directly.

## Automated evidence

- Strict TypeScript typecheck passed for the application configuration.
- Seven deterministic Vitest tests passed for event-envelope parsing, explicit
  gap controls, stable-ID ordering/deduplication, bounded state, filters, local
  day boundaries, and safe API-error mapping.
- The Angular production build passed with a 336.74 kB raw initial bundle and an
  estimated 94.43 kB initial transfer. Every feature page is a lazy chunk.
- Production dependency audit reported zero vulnerabilities. Dev-only build
  dependencies are not copied into the Nginx runtime image.
- The self-contained Python suite remained green at 87 passed and 9 skipped.
- The complete MongoDB 8, Redis 8, and MinIO suite passed 96 tests.
- Ruff and Python bytecode compilation passed after adding the production
  WebSocket runtime dependency.
- Compose validation and both production images built successfully.
- Nginx container health passed; root and deep SPA routes returned the compiled
  application, /api requests reached FastAPI on the same origin, and the shell
  returned no-store plus CSP, frame-deny, MIME, referrer, and permissions headers.
- With authentication enabled, missing and incorrect keys returned 401, the
  temporary ADMIN key resolved the expected principal, and a browser-style
  first-frame authenticated WebSocket through Nginx received
  system.realtime.ready.
- With authentication disabled, the development principal and WebSocket ready
  control were also verified through the same gateway.

The one Python warning is an upstream FastAPI/Starlette TestClient deprecation
for httpx and is unchanged from earlier milestones.

## Security and failure semantics

The API key is tab-scoped, never sent in a URL, and is cleared on logout, failed
validation, REST 401, or WebSocket authorization failure. Browser storage failure
degrades to in-memory use for the current page rather than leaking the key.

Frontend role checks hide unavailable actions, while backend RBAC remains the
authoritative enforcement point. RTSP credentials are accepted only in the
camera-create form, cleared after the request, and never returned by the API.

Realtime is best effort. The client reconnects with capped exponential backoff,
retains only the stable last event ID, and treats a gap as a mandatory REST
reconciliation signal. A realtime failure does not erase the last successful
REST view.

## Operational limits

- Full npm audit reports eleven dev-only advisories inherited through the Angular
  CLI/build graph. The shipped dependency audit is zero because the multi-stage
  image contains only Nginx and compiled assets; build jobs must remain isolated
  while upstream fixes are pending.
- Dashboard counters are based on a bounded event page and show a plus sign when
  more records exist; exact high-volume aggregation needs a backend statistics
  endpoint.
- Camera cards show latest state rather than video. Live monitor transport and
  overlays are separate work.
- Media object keys cannot be rendered until a signed URL/download API exists.
- Alert and camera-health topics are not on the realtime fan-out yet, so those
  views use event-triggered refresh or polling.
- The in-app browser runtime exposed no browser instance in this environment, so
  visual interaction automation was unavailable. Compilation, unit tests,
  production container rendering paths, HTTP headers, REST and WebSocket protocol
  smoke tests were completed; manual cross-browser visual QA remains required
  before an external production release.
