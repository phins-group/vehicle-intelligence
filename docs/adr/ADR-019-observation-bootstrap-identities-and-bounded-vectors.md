# ADR-019: Bootstrap identity per observation and bound vector search

## Status

Accepted — 2026-08-10.

## Context

A local camera track, a recognized plate, and a global physical vehicle are
different concepts. Treating equal plate text as identity would silently merge
cloned, misread, changed, or obscured plates. Embeddings are large, model-specific
evidence and a Python-side collection scan cannot scale safely.

## Decision

The event worker creates one deterministic bootstrap `VehicleIdentity` and one
immutable `VehicleFingerprint` per canonical event. It links the event by
`vehicleId`, but does not merge on plate. Identity aggregates keep only bounded
aliases and counters; event history stays in `vehicle_events`.

Visual vectors live in `vehicle_embeddings`. Fingerprints hold only a reference
and the exact model name/version/dimension/hash. `VectorRepository.search`
requires a pre-filtered, explicitly bounded candidate-ID set and rejects model or
dimension mismatch. The TorchScript provider verifies the configured checkpoint
hash and emits normalized, dimension-checked vectors.

## Consequences

- Bootstrap creates more identity rows before ReID, but never invents certainty.
- Candidate generation must use topology/time before visual similarity.
- Model upgrades cannot accidentally compare incompatible vector spaces.
- A dedicated vector engine can replace Mongo without changing domain code.
