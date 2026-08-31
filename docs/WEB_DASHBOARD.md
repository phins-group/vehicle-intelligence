# Angular Operator Dashboard

## Scope

The web application in apps/web is the Phase 2 operator-console foundation. It
uses Angular 22, standalone lazy routes, TypeScript 6 strict mode, RxJS for stream
coordination, zoneless change detection, and Angular signals for local view state. It renders only API data;
there is no demo-data fallback in production code.

Implemented routes:

| Route | Capability |
|---|---|
| /dashboard | Today counters, seven-hour activity, camera availability, recent events |
| /events | Combined filters, cursor pagination, live prepend, event evidence drawer |
| /vehicle-search | Exact canonical plate search, loaded-evidence summary, cursor timeline |
| /ocr-review | OPERATOR/ADMIN queue, signed evidence, revisioned confirmation/correction |
| /dataset-review | OPERATOR/ADMIN detector-label queue, bbox editor, revision history, immutable promotion |
| /datasets | Immutable catalog, paged sample/bbox viewer, lineage evidence, private Hugging Face sync |
| /cameras | Latest health, create, enable/disable, and connection test |
| /live-monitor | Low-rate exact-frame preview with configurable AI overlays |
| /alerts | Cursor listing, filters, acknowledge, and resolve workflow |
| /watchlists | Plate/type/status filters, validity lifecycle, ADMIN create/edit/delete |
| /rules | Priority-ordered rule cards and allowlisted ADMIN rule builder |
| /model-quality | Bounded model-version quality, daily trend, and retraining feedback state |
| /system-health | API, realtime, and per-camera health with ten-second refresh |
| /login | OIDC Authorization Code + PKCE or API-key fallback with safe return-route handling |

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

The client reads `/api/system/health` and the public, secret-free
`/api/auth/config` contract. When authentication is disabled, it resolves the
development principal through `/api/auth/me`. An `api_key` deployment accepts a
manually supplied key. An `oidc` deployment redirects through Authorization Code
with PKCE S256, validates the callback state and age, exchanges the one-time code,
and asks `/api/auth/me` to authorize the resulting access token.

The raw key:

- is stored only in sessionStorage for the current tab session;
- is attached as a Bearer header only to same-origin /api requests;
- is sent in the WebSocket authenticate first frame, never in its URL;
- is cleared after logout, failed validation, or WebSocket 4401/4403;
- is never rendered after login or written to application logs.

OIDC access tokens are kept in memory only and disappear on reload or logout.
Only the short-lived PKCE verifier/state transaction is kept in `sessionStorage`.
The IdP must register the exact callback path and allow token-endpoint CORS only
for the console origin.

The UI hides mutation controls according to the backend role matrix, but this is
only a usability measure. FastAPI remains the authorization boundary:

| Role | Dashboard behavior |
|---|---|
| VIEWER | Read-only pages |
| OPERATOR | Read, camera connection test, alert workflow, OCR review, and detector-label review |
| ADMIN | All operator actions plus camera/watchlist/rule mutations, dataset promotion, and private Hub sync |

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

## Human detector-dataset review

The `/dataset-review` lazy route is deliberately separate from OCR review. It
loads immutable first-party detector sources, filters the queue by state and
reason, fetches image evidence with the authenticated API client, and draws SVG
boxes in source-image pixel coordinates. Operators can confirm an unchanged
model suggestion, save corrected `license_plate` boxes, mark a verified hard
negative, or reject an unusable image with a required explanation.

Every write carries the last observed revision. FastAPI revalidates box bounds
and action semantics, returns `409` for concurrent edits, and writes a new
no-overwrite decision revision bound to the source manifest, queue checksum,
image checksum, actor, and UTC time. The source queue and original image remain
unchanged. Images are exposed as short-lived in-memory Blob URLs in the browser;
the Nginx policy allows `blob:` only for `img-src`, while scripts remain limited
to `'self'`.

ADMIN can start an asynchronous promotion that creates a new immutable source
ID. Confirmed/corrected/negative decisions are included, rejected samples are
excluded, and unresolved samples remain in the next review queue. Promotion
does not mutate the parent source and emits a verifiable manifest plus review
evidence. See [Detector training](DETECTOR_TRAINING.md#human-review-ui-for-detector-labels)
for the source layout and operating procedure.

## Dataset management and private Hub sync

The `/datasets` lazy route is the post-promotion operations screen. It shows
source versions and parent lineage, exact source/export manifest hashes,
sample/annotation/negative counts, review readiness, COCO split counts, and the
latest durable synchronization job. It does not upload the mutable review
workspace or raw source tree directly: the API first creates or reuses an
immutable COCO export bound to the selected source manifest and verifies every
recorded byte.

The same page contains **Xem mẫu Dataset**, a read-only gallery for the selected
source version. It loads 12 records at a time using a source-manifest-bound
cursor, can filter positive/negative and day/night samples, and overlays the
canonical `license_plate` boxes in source-image coordinates. Selecting a card
shows image identity, split, camera/group evidence, review state, checksum, and
exact bbox values. This viewer is for quality inspection only; edits still go
through `/dataset-review` and an immutable promotion.

Sample images are fetched as authenticated Blob responses and exposed only by
source ID plus SHA-256, never by a filesystem path. The backend checks manifest
membership, file size, checksum, image decoding, pixel limits, and bbox bounds
before returning metadata or image bytes. The browser revokes object URLs when
the source/filter changes or the component closes.

Only ADMIN sees the sync action. A restricted first-party source also requires
an explicit per-request confirmation and the server-side
`restricted_private_sync_enabled` policy. The UI exposes whether Hub support,
the destination repository, server policy, and credentials are available, but
never receives the credential itself. Job polling renders `QUEUED`,
`PREPARING_EXPORT`, `UPLOADING`, `COMPLETED`, or `FAILED`; a successful job links
to the exact private Hub commit returned by the backend.

FastAPI remains authoritative for validation, RBAC, idempotency, audit ordering,
source/export checksum binding, and the remote-private check. See
[Detector training](DETECTOR_TRAINING.md#dataset-catalog-and-private-hub-sync-after-promotion)
and [PHINS governance](PHINS_DATASET_GOVERNANCE.md#reviewed-first-party-versions-and-private-hub-processing).

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

Use an Angular-supported Node version. The repository pins Node 24.15.0 in
apps/web/.nvmrc.

    cd apps/web
    nvm use
    npm ci
    npm start

The development proxy forwards /api and /ws to 127.0.0.1:8000. Start the API
separately; no CORS configuration is needed.

Useful checks:

    npm run lint
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

- Both the production and complete web lockfile graphs pass `npm audit` without
  exceptions. The final Nginx image contains no Node build toolchain; CI still
  audits the complete builder graph before compiling it.
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
- The backend accepts API-key or OIDC/JWKS Bearer credentials. The console now
  implements Authorization Code + PKCE for public OIDC clients and retains the
  access token in memory. It does not provide a server-side BFF/cookie session;
  deployments requiring refresh without a redirect should add that as a separate
  reviewed trust boundary.
- Browser visual automation was unavailable in the current execution
  environment; compiler, deterministic tests, container and HTTP/proxy smoke
  checks remain the acceptance evidence.
