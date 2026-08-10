# Phase 2 Human OCR Review Acceptance Record

## Decision

The human OCR review and dataset-feedback milestone passed engineering
acceptance on 2026-08-09. This accepts immutable AI predictions, revisioned
human confirmation/correction, an indexed effective plate, role-protected API
and Angular review queue, audit evidence, and idempotent training samples that
reference an existing plate crop.

This milestone does **not** accept manual plate assignment when no OCR
prediction exists, transactional writes across all review collections,
automatic historical rule replay, dataset export, fuzzy search, or cross-camera
vehicle identity.

## Accepted flow

```text
NEEDS_REVIEW event + durable plate crop
  -> OPERATOR/ADMIN opens /ocr-review
  -> compare AI prediction with signed evidence
  -> normalize and submit expected review revision
  -> atomically amend event to schema v2
  -> retain prediction + store review + index final plate
  -> append audit record
  -> create deterministic dataset sample
```

A correction never overwrites the raw or normalized AI prediction. The event's
`plate.final` becomes the effective search value, while the dataset sample
retains prediction, human label, reason, model trace, and an object-storage key
instead of image bytes.

## Implemented

- `PlateReview` and `DatasetSample` domain models with bounded, versioned
  serialization and backward-compatible schema-v1 reads.
- Atomic expected-revision updates in MongoDB and in-memory repositories, plus
  durable atomic JSONL rewriting for Mongo-disabled development.
- `PUT /api/events/{event_id}/plate-review` with normalization, validation,
  idempotent retries, `409` conflict handling, no-store responses, and
  OPERATOR/ADMIN authorization.
- `GET /api/dataset-samples` with bounded cursor pagination and indexed type,
  status, and reason filters.
- Immutable prediction, current review, and final effective plate fields;
  exact search supports indexed v2 final plates and an indexed legacy fallback.
- Actor audit for confirmation and correction, without logging credentials or
  media content.
- Deterministic feedback sample IDs and a media-inspection port. A sample is
  created only when the referenced plate-crop object actually exists.
- Lazy Angular `/ocr-review` queue with signed evidence, AI/human comparison,
  revision-aware form, conflict refresh, realtime queue insertion, and
  read-only handling outside authorized roles.

## Tested

- The complete suite passed **115 Python tests** against real MongoDB 8, Redis
  8, and MinIO. Coverage includes correction, confirmation, immutable
  prediction, idempotent retry, stale conflict, validation, missing-media
  behavior, indexed search, audit, cursor filtering, repository indexes,
  serialization, JSONL reload, and MinIO existence checks.
- Ruff passed for `src`, `tests`, and `scripts`.
- Strict TypeScript typecheck passed. All **24 Vitest tests** in five files
  passed, including final-plate and review-revision presentation semantics.
- The Angular production build passed at **357.20 kB raw / 97.77 kB estimated**
  initial transfer. The OCR review page remains a lazy **11.78 kB** chunk.
- Production dependency audit reported **zero vulnerabilities**. The complete
  build-only dependency graph reports seven moderate and four high Angular
  tooling advisories; Node/npm are not present in the Nginx runtime image.
- Compose built the Python 3.12 API and Node 24.12/Nginx web images. MongoDB,
  Redis, MinIO, API, and web became healthy/running; `/ocr-review` and
  `/api/system/health` returned `200` through Nginx with no-store/security
  headers, and health reported human review available.
- A scoped end-to-end smoke event with a real 518-byte MinIO JPEG was corrected
  from `51H-123.45` to `51H-123.46`. The API retained the prediction, returned
  the final label, exposed signed media, created one `HUMAN_CORRECTION` sample
  with model trace, returned the same sample on an identical retry, rejected a
  stale different review with `409`, found the event by its final plate, and
  wrote exactly one `PLATE_CORRECTED` audit record.
- The exact smoke event, sample, audit record, and MinIO object were removed.
  Post-cleanup checks found zero scoped documents and `NoSuchKey` for the media.

The unchanged Starlette TestClient/httpx compatibility warning does not affect
the suite. The unsupported host Node 23 runtime only disables an optional
Angular compiler cache; the reproducible Compose build uses Node 24.12.

## Consistency and security semantics

The client submits `expectedRevision`; the event repository compares it in the
same atomic mutation that writes the review. A different stale submission
cannot silently replace another operator's work. An identical network retry is
recognized as already applied and reuses the deterministic dataset sample.

VIEWER can inspect event evidence but cannot submit a review or enumerate
dataset samples. OPERATOR and ADMIN receive `REVIEW_PLATES`. The server derives
reviewer identity from the authenticated principal, normalizes input itself,
and never trusts client-supplied actor or final-state fields.

## Known limitations

- Event amendment, dataset sample, and audit are separate writes on standalone
  MongoDB. Strict all-or-nothing behavior requires a replica-set transaction or
  transactional outbox. A retry repairs a missing deterministic sample, but is
  not a substitute for cross-collection atomicity.
- The workflow requires an existing OCR prediction; it cannot yet assign a
  plate to an event whose `plate` is null.
- Corrections do not replay historical rules, actions, alerts, or barrier
  decisions automatically.
- Dataset export, labeling lifecycle transitions, and retention pinning for
  referenced crops remain future work. Storage retention must not delete a crop
  before its accepted sample is exported.
- Review amendments have no dedicated realtime update topic. Other open review
  consoles reconcile by refreshing after a conflict or queue reload.
- Fuzzy plate search and cross-camera identity/ReID remain intentionally out of
  scope.
- Manual cross-browser visual and keyboard QA remains required before an
  external production release; automated UI, build, route, API, RBAC, and
  container checks passed.

## Next

The next UI milestone is the live monitor with optional vehicle/plate boxes,
track IDs, recognized plate, direction, confidence, ROI, and trigger-line
overlays. It should consume bounded metadata rather than stream raw frames over
the event WebSocket.

Strict transactional audit/outbox, dataset export/retention coordination, and
production connection/load/soak validation remain platform-hardening work.
