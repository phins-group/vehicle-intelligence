# Angular Operator Dashboard

## Scope

The web application in apps/web is the Phase 2 operator-console foundation. It
uses Angular 21, standalone lazy routes, TypeScript strict mode, RxJS for stream
coordination, and Angular signals for local view state. It renders only API data;
there is no demo-data fallback in production code.

Implemented routes:

| Route | Capability |
|---|---|
| /dashboard | Today counters, seven-hour activity, camera availability, recent events |
| /events | Combined filters, cursor pagination, live prepend, event evidence drawer |
| /vehicle-search | Exact canonical plate search, loaded-evidence summary, cursor timeline |
| /ocr-review | OPERATOR/ADMIN queue, signed evidence, revisioned confirmation/correction |
| /cameras | Latest health, create, enable/disable, and connection test |
| /live-monitor | Low-rate exact-frame preview with configurable AI overlays |
| /alerts | Cursor listing, filters, acknowledge, and resolve workflow |
| /watchlists | Plate/type/status filters, validity lifecycle, ADMIN create/edit/delete |
| /rules | Priority-ordered rule cards and allowlisted ADMIN rule builder |
| /model-quality | Bounded model-version quality, daily trend, and retraining feedback state |
| /system-health | API, realtime, and per-camera health with ten-second refresh |
| /login | API-key authentication and safe return-route handling |

Logical vehicle identity and cross-camera journeys are available through the
vehicle detail route. Automatic ReID mutation and full-rate HLS/WebRTC video are
intentionally not claimed by the current console.

## Runtime boundaries

    Browser
      -> same-origin Nginx
          -> static Angular assets
          -> /api/* to FastAPI
          -> /ws/* to FastAPI WebSocket

REST remains canonical. WebSocket delivery is a low-latency invalidation/event
signal. The client deduplicates by VehicleEvent._id, keeps bounded in-memory
lists, and reloads the MongoDB-backed REST page whenever the server emits a
system.realtime.gap control.

The event WebSocket never streams raw frames. Live monitoring instead polls
authenticated metadata and its exact low-rate JPEG sequence from dedicated
endpoints; camera cards otherwise show only latest telemetry.

## Authentication and RBAC

The client first reads the public /api/system/health endpoint. When backend
authentication is disabled, it resolves the development principal through
/api/auth/me. When authentication is enabled, the operator supplies an API key.

The raw key:

- is stored only in sessionStorage for the current tab session;
- is attached as a Bearer header only to same-origin /api requests;
- is sent in the WebSocket authenticate first frame, never in its URL;
- is cleared after logout, failed validation, or WebSocket 4401/4403;
- is never rendered after login or written to application logs.

The UI hides mutation controls according to the backend role matrix, but this is
only a usability measure. FastAPI remains the authorization boundary:

| Role | Dashboard behavior |
|---|---|
| VIEWER | Read-only pages |
| OPERATOR | Read, camera connection test, alert workflow, OCR review/dataset feedback |
| ADMIN | All operator actions plus camera and watchlist/rule mutations |

## Policy authoring

The watchlist page computes `ACTIVE`, `SCHEDULED`, `EXPIRED`, and `DISABLED`
from the server entry and the browser clock. Local datetime fields are converted
to timezone-aware UTC ISO-8601 values before submission. Vietnamese plate
normalization remains authoritative in the backend.

The rule page never accepts code, expressions, or a raw rule JSON document. It
builds requests from the backend's allowlisted fields, operators, action types,
HTTP methods, and alert severities. `IN`/`NOT_IN` text is converted to a bounded
non-empty list; external action URLs must use HTTP(S) and cannot contain URL
credentials. The backend repeats every validation and its host allowlist remains
authoritative.

Edits preserve the last server `revision`; concurrent updates return `409`
instead of silently overwriting. Rule action IDs are preserved during editing so
the event/rule/action idempotency key remains stable. Deletes require an explicit
in-app confirmation naming the resource. All successful policy mutations are
audited by FastAPI.

## Plate-centric investigation

`/vehicle-search?plate=...` is a shareable, read-only investigation route. The
API normalizes the Vietnamese plate and performs an exact indexed query on
`plate.normalized`; it does not OCR images again or scan all events. Results are
cursor-paginated newest-first at the API, while the view renders its currently
loaded observations chronologically from old to new.

Every summary label says “loaded” because older cursor pages may still exist.
The browser retains at most 500 events and deduplicates by canonical event ID.
The view exposes camera, direction, type/color evidence, OCR confidence,
assigned/null `vehicleId` state, object keys, and event/track traceability. Event
detail links can deep-link directly into the same search.

This is not a logical `Vehicle` detail page. A plate is a strong signal but may
be cloned, reassigned, or read incorrectly, so the UI does not merge tracks or
claim that every matching observation is one physical vehicle. Fuzzy OCR search,
visual embeddings, topology constraints, and cross-camera journeys remain
separate capabilities.

## Human OCR review

The `/ocr-review` lazy route queries the indexed `NEEDS_REVIEW` event queue and
uses the existing signed-media component for snapshot/plate evidence. The form
submits the last observed review revision, so another operator's newer decision
produces `409` and refreshes the drawer instead of being overwritten.

The UI displays AI raw/normalized text, confidence, observation count and media
alongside the editable final plate. Successful review removes the event from the
local queue and reports whether a deterministic dataset sample was created.
There is no browser-side normalization authority, image upload, object-key
signing, or direct database access. The backend derives the actor from the
authenticated principal and preserves the prediction.

## Media evidence

The event and plate-investigation drawers share one evidence viewer. It requests
`/api/events/{eventId}/media` only while the selected drawer exists. FastAPI
authorizes the event and returns short-lived URLs for available snapshot,
vehicle crop, plate crop, and optional clip references. The browser never asks
the server to sign an object key directly.

Images use product-specific alternative text and open their original URL with
`noopener noreferrer`; clips use metadata-only preload. Missing storage objects
remain visible as `MISSING` evidence gaps with their durable key. Request errors
have an explicit retry state. The viewer refreshes URLs thirty seconds before
expiry, guards against stale responses when selection changes, and clears its
timer/URL state when destroyed. URLs are not written to browser storage.

## Live monitor

`/live-monitor?camera=...` is a lazy, shareable route for enabled cameras. The
browser polls state only while the tab is visible, then fetches the exact JPEG
sequence as an authenticated Blob. It verifies the response sequence before
swapping frames, revokes the previous object URL, and rejects stale responses
after camera changes.

Source-coordinate SVG overlays remain aligned through the frame `viewBox`.
Operators may independently toggle vehicle boxes, plate boxes, track IDs, plate
text, direction, confidence, ROI, and crossing line. The screen presents
`WAITING`, `LIVE`, `STALE`, `OFFLINE`, and `DISABLED` states explicitly. It does
not open the RTSP URL, persist images, or claim to be continuous video. See
[Live Monitor](LIVE_MONITOR.md) for transport and buffer limits.

## ONVIF camera discovery

The camera page lets OPERATOR/ADMIN users run one bounded local-network ONVIF
scan. Results are temporary and show only safe device-service metadata. ADMIN
users can use a result to prefill camera ID, name, location, and provenance, but
must separately enter the RTSP URL; no discovery credential is retained or
rendered. Camera creation remains the backend authorization and encryption
boundary. See [ONVIF and multi-camera ingress](ONVIF_AND_MULTI_CAMERA_INGRESS.md).

## Realtime behavior

RealtimeService owns one WebSocket while the authenticated shell exists. It:

1. connects to /ws/events on the current origin;
2. authenticates in the first frame when a key exists;
3. records the last stable event ID for bounded replay;
4. reconnects with capped exponential backoff;
5. emits typed vehicle events through RxJS;
6. requests REST reconciliation after an explicit gap;
7. stops reconnecting on authorization failure.

Dashboard and alert refreshes are rate-limited with RxJS audit windows. The
event explorer can prepend matching events immediately and suppress duplicates.

## Model quality

`/model-quality` requests a timezone-aware, bounded report for 7, 30, or 90
days. It presents OCR success, unknown-plate rate, correction rate, average plate
confidence, UTC daily event volume, model name/version/hash slices, and dataset
export states. Metric denominators remain visible in the UI. The browser does
not scan events, derive model identity, access object keys, start an export, or
promote a model; MongoDB aggregation and offline tooling remain backend/operator
boundaries.

## Local development

Use an Angular-supported Node version. The repository pins Node 24.12.0 in
apps/web/.nvmrc.

    cd apps/web
    nvm use
    npm ci
    npm start

The development proxy forwards /api and /ws to 127.0.0.1:8000. Start the API
separately; no CORS configuration is needed.

Useful checks:

    npm run typecheck
    npm test
    npm run build
    npm audit --omit=dev

## Container deployment

The multi-stage image builds Angular under pinned Node/npm versions and copies
only the compiled browser directory into Nginx. The build toolchain and its dev
dependencies are absent from the runtime image.

    docker compose up -d mongodb redis api web
    open http://localhost:4200

Nginx provides SPA fallback, immutable caching for hashed assets, no-store for
the shell document, same-origin API/WebSocket proxying, a health endpoint, and
baseline browser security headers. Its dedicated `/vehicle-media/` path proxies
signed GET/HEAD requests to MinIO with the original Host and without access-log
query leakage. Image/media CSP therefore remains same-origin. TLS must terminate
at Nginx or an upstream ingress in non-local environments.

## Operational limitations

- The complete development audit currently reports eleven advisories in the
  Angular CLI/build-only dependency graph. The production dependency audit is
  zero and the final Nginx image contains no Node build toolchain. CI builders
  should remain isolated from untrusted projects/inputs while upstream packages
  are updated.
- The dashboard summary reads at most 200 events from the current local day and
  explicitly displays a plus sign when a continuation cursor exists; it does not
  claim an exact aggregate beyond the loaded page.
- Signed media requires a browser-reachable public gateway endpoint and a
  Host-preserving proxy for the configured bucket path. The local filesystem
  adapter remains intentionally unavailable through HTTP, and URL revocation is
  bounded by the configured short expiry.
- Watchlist and rule list APIs are bounded at 200 items and have no cursor yet;
  their UI filters operate within that loaded set and do not claim a complete
  count beyond the API limit.
- Plate history has cursor pagination but the browser deliberately stops after
  500 loaded events. It supports exact canonical matching only; indexed fuzzy
  candidates require a separate backend search design.
- Alert-created and camera-health realtime topics are not published yet; those
  pages refresh from REST after vehicle events or on their polling interval.
- Human-review amendments are not broadcast as a separate realtime topic yet;
  other consoles reconcile through queue refresh. Reviewed events without an
  available plate crop cannot produce an OCR dataset sample. `READY`,
  `EXPORTING`, and `EXPORT_FAILED` samples automatically pin their crop and
  source event until a verified export reaches `EXPORTED`.
- Live preview is intentionally low-rate and best effort. Its frame ring is
  API-process-local, so state/frame requests need sticky routing until a shared
  bounded cache or dedicated HLS/WebRTC gateway is introduced.
- The backend accepts API-key or OIDC/JWKS Bearer credentials. The console keeps
  a manually supplied bearer token only in tab-scoped session storage; it does
  not implement an OIDC authorization-code/PKCE redirect or server cookie session.
- Browser visual automation was unavailable in the current execution
  environment; compiler, deterministic tests, container and HTTP/proxy smoke
  checks remain the acceptance evidence.
