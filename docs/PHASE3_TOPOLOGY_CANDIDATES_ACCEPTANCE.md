# Phase 3 Topology and Candidate Acceptance

Accepted on 2026-08-10.

## Implemented

- Directed, revisioned `CameraTopologyEdge` domain model with self-loop/window
  validation and unique `(fromCameraId, toCameraId)` persistence.
- In-memory and MongoDB repositories with inbound/outbound indexes, optimistic
  updates, bounded lists, and no implicit reverse edge.
- Cross-camera candidate generator that derives a prior-time window from each
  enabled inbound edge and uses the indexed fingerprint camera/time query.
- Hard configuration limits for edges, observations per edge, and total results;
  candidates are ranked by calibrated proximity to typical travel time.
- ADMIN-only, transactional/audited topology CRUD plus read-only candidate API.

## Tested

- Unit coverage proves self-loop/window rejection, directional uniqueness,
  optimistic conflicts, time-window exclusion, wrong-camera exclusion, reverse
  direction rejection, ranking, and hard result caps.
- Real MongoDB replica-set integration verifies indexes, CRUD revisions, bounded
  camera/time retrieval, and candidate output.
- API integration verifies valid camera references, CRUD, stale revision `409`,
  and candidate response.
- Full real-service suite: **168 passed, 0 skipped**; Ruff passes.

## Known limitations

- Travel windows are operator-configured rather than learned from telemetry.
- This milestone returns candidates and time evidence only; multi-signal ReID
  scoring and reviewed identity mutation are the next milestone.

## Next

Versioned ReID scoring plus transactional human merge/split review.
