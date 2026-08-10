# Phase 3 Identity Foundation Acceptance

Accepted on 2026-08-10.

## Implemented

- Model-agnostic `VehicleIdentity`, immutable `VehicleFingerprint`, versioned
  `EmbeddingReference`/`EmbeddingVector`, repository, vector-search, and embedding
  provider ports.
- Safe deterministic identity bootstrap per event. Plate text remains a signal;
  two equal plates do not silently become one physical vehicle.
- Bounded identity aggregates with at most 16 plate aliases and event history
  retained only in indexed `vehicle_events.vehicleId` queries.
- Separate MongoDB `vehicles`, `vehicle_fingerprints`, and
  `vehicle_embeddings` collections with access-pattern indexes and idempotent
  source-event registration.
- Candidate-only vector search with strict model name/version/dimension matching,
  maximum candidate limits, normalized cosine scoring, and no full scan.
- Real TorchScript embedding adapter with SHA-256 checkpoint verification,
  image validation, L2 normalization, and output-dimension enforcement.
- Event-worker identity post-processing plus REST identity/fingerprint reads.

## Tested

- Unit tests cover idempotent bootstrap, same-plate separation, bounded/model-safe
  vector queries, real temporary TorchScript inference, normalization, and hash
  rejection.
- Real MongoDB replica-set integration verifies identity/fingerprint persistence,
  event linking, indexes, and candidate-restricted vector results.
- A real Redis event-worker acceptance created one linked event, one identity,
  and one fingerprint without an unbounded event array.
- Full suite with real MongoDB, Redis, and MinIO: **162 passed, 0 skipped**.
- Ruff: all source, tests, and scripts pass.

## Known limitations

- Identity bootstrap intentionally does not merge observations. Topology/time
  candidate generation and scored merge decisions belong to the next milestones.
- Embedding extraction is disabled by default until an operator supplies a
  verified checkpoint; the project does not ship or fabricate model weights.

## Next

Camera topology and bounded travel-time candidate generation.
