# ADR-020: Filter ReID candidates through directed camera topology

## Status

Accepted — 2026-08-10.

## Decision

Camera topology is a revisioned directed graph. Every edge declares minimum,
typical, and maximum travel time. A new fingerprint at camera B searches only
enabled inbound edges A→B, then queries fingerprints at A within the derived
indexed time window. Reverse travel requires its own edge.

Per-query edges, candidates per edge, and total returned candidates are
configuration-bounded. Visual vectors are not touched until this pre-filter has
produced explicit IDs.

## Consequences

- Physically impossible observations are rejected before expensive scoring.
- One-way routes and asymmetric travel times are represented correctly.
- Missing topology produces no candidate rather than an unsafe global scan.
- Operators must maintain the graph and tune windows from measured travel data.
