# ADR-013: Event-Scoped Short-Lived Media Access

- Status: Accepted
- Date: 2026-08-09

## Context

Vehicle events persist object keys rather than image/video bytes. Operators need
to inspect snapshots, vehicle crops, plate crops, and optional clips, but making
the MinIO bucket public would bypass API authorization. Accepting an object key
from the browser and signing it would also create an arbitrary bucket-read
primitive. Internal Compose hostnames are not necessarily reachable from a
browser, and historical event references can outlive the underlying object.

## Decision

Expose `GET /api/events/{eventId}/media` under the existing `READ_PLATFORM`
permission. The application first loads the canonical event, takes media keys
only from that event, validates them as relative object keys, checks object
existence through a replaceable `MediaUrlSigner`, and returns fixed-lifetime
presigned GET URLs. There is no API that accepts an arbitrary key.

Use separate internal and browser-reachable MinIO endpoints. The internal client
performs object checks; the public-endpoint client signs the URL without making a
network request. In the bundled deployment, the browser endpoint is the same
Nginx origin: a dedicated `/vehicle-media/` location preserves the signed Host,
proxies only GET/HEAD to MinIO, disables access logging, and keeps CSP media
sources at `'self'`. Cap configured lifetime at one hour and default it to five
minutes. Mark the API response `no-store, private`; never log or persist signed
URLs. Represent a deleted object as `MISSING` while preserving its durable key.

Angular requests media only while an event drawer exists, opens originals with
no referrer, automatically refreshes before expiry, and discards URLs when the
drawer is destroyed.

## Consequences

- VIEWER, OPERATOR, and ADMIN can inspect evidence without public bucket policy.
- Possession of a signed URL permits read access until expiry, so TLS and short
  lifetimes remain mandatory.
- Revoking API credentials does not invalidate an already issued URL; urgent
  revocation requires object/key rotation or storage-layer controls.
- Deployments must configure `public_endpoint` to the operator gateway hostname
  and route the signed bucket path to object storage while preserving `Host`;
  the internal service name is not returned accidentally.
- Local-filesystem media remains writable for the Phase 1 CLI but has no HTTP
  exposure in this milestone; its API reports media access unavailable.
- Retention cleanup can delete an object independently; the evidence UI reports
  that gap rather than removing or rewriting the historical event.
