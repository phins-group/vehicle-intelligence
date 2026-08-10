# Phase 2 Vehicle Search UI Acceptance Record

## Decision

The plate-centric vehicle-search milestone passed engineering acceptance on
2026-08-09. This accepts exact Vietnamese plate normalization, indexed event
history lookup, stable cursor pagination, a shareable Angular search route,
loaded-window summaries, chronological evidence display, event inspection, and
role-protected same-origin access.

This milestone does **not** assert that a plate is a global vehicle identity. It
does not accept fuzzy search, cross-camera ReID, vehicle journeys, signed media
delivery, human OCR correction, or browser connection/load/soak limits.

## Accepted flow

```text
Angular /vehicle-search?plate=51H98765
  -> FastAPI normalization to 51H-987.65
  -> indexed MongoDB query by plate.normalized + occurredAt
  -> opaque cursor pages, newest first
  -> loaded-window summary + chronological evidence timeline
  -> optional event evidence drawer
```

The API returns observations, not an inferred identity. The UI therefore labels
counts and dates as belonging to the currently loaded evidence, exposes distinct
logical `vehicleId` signals where present, and keeps unresolved observations
visible instead of silently merging them.

## Automated evidence

- The complete real MongoDB 8, Redis 8, and MinIO suite passed 97 Python tests.
  The new integration coverage verifies canonical exact search, stable cursor
  ordering, a second page, invalid-cursor rejection, and VIEWER access. Ruff and
  Python bytecode compilation also passed.
- Strict application/spec TypeScript typecheck passed. Eighteen deterministic
  Vitest tests passed, including chronological ordering, bounded loaded-summary
  semantics, deterministic ties, and the empty state.
- The Angular production build passed at 352.56 kB raw / 96.99 kB estimated
  initial transfer. Vehicle search remains a lazy chunk at 17.14 kB raw /
  5.30 kB estimated transfer.
- Production dependency audit reported zero vulnerabilities. The complete
  build-only dependency graph reports seven moderate and four high transitive
  advisories through Angular tooling; Node/npm are absent from the Nginx runtime
  image.
- Compose validation and the current API/Nginx images built successfully. The
  API, web, MongoDB, Redis, and MinIO services became healthy/running.
- Three scoped MongoDB events proved same-origin search through Nginx. Input
  `51H98765` normalized to `51H-987.65`; page one returned the two newest events,
  its opaque cursor returned the oldest event on page two, an invalid cursor
  returned `400`, and `/vehicle-search?plate=51H98765` returned the SPA with
  `200`.
- With authentication temporarily enabled, missing and invalid credentials
  returned `401`; VIEWER, OPERATOR, and ADMIN each read the search endpoint with
  `200`, and `/api/auth/me` resolved all three expected roles. The configuration
  guard also correctly rejected an auth setup without an active ADMIN.
- Compose was restored to development `auth=disabled`, the Nginx search proxy
  returned `200`, and the three exact smoke event IDs were deleted. MongoDB
  confirmed zero scoped records remained.

One unchanged warning comes from FastAPI/Starlette's current TestClient use of
httpx. The host's unsupported odd-numbered Node 23 runtime also disabled an
optional Angular compiler cache, but did not alter or invalidate build output;
the Compose build uses the project-supported Node image.

## Search and consistency semantics

Search is exact after canonical Vietnamese plate normalization. It reuses the
event repository and its compound plate/time index instead of loading documents
for Python-side matching. Cursor state contains the stable event sort boundary,
and malformed cursor input is rejected rather than interpreted loosely.

The browser retains event pages newest-first for cursor merging, then derives an
ascending timeline from the loaded set. A request-generation guard prevents a
slower prior query from replacing a newer search. Refresh clears the cursor and
loaded observations; load-more merges by event ID.

The browser caps one investigation at 500 loaded observations. Metrics, date
bounds, camera counts, common attributes, confidence, and identity-signal counts
are explicitly scoped to that loaded window. This avoids both unbounded memory
growth and misleading claims about the complete history.

## Operational limits

- Fuzzy plate search is intentionally absent. Adding it requires a bounded,
  indexed candidate representation or search adapter; a collection scan is not
  acceptable.
- A repeated plate may be cloned, misread, or reassigned. Cross-camera vehicle
  identity and journeys remain Phase 3 work based on plate, visual embeddings,
  attributes, topology, and travel time.
- Media cards currently expose durable object keys as evidence. Signed media URL
  delivery and expiry handling remain separate work.
- The investigation page is request-driven and does not merge realtime events.
- The in-app browser runtime exposed no browser instance in this environment.
  Compiler checks, deterministic tests, production builds, container routes,
  REST semantics, RBAC, and cleanup were verified; manual cross-browser visual
  and keyboard QA remains required before an external production release.
