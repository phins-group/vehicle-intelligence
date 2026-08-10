# ADR-022: Derive journeys from canonical vehicle events

- Status: accepted
- Date: 2026-08-10

## Context

Logical vehicle history must support chronological timelines and topology-aware
travel analysis. Embedding an event array in `vehicles` would grow without bound,
duplicate canonical facts, and make reviewed merge/split operations fragile.

## Decision

Query `vehicle_events` by the indexed `(vehicleId, occurredAt)` access pattern and
derive journey observations and consecutive segments at read time. Annotate a
segment only with the exact directed camera-topology edge. Bound every read and
return an explicit truncation flag.

## Consequences

- Canonical events remain the only history source of truth.
- Reviewed identity changes become visible immediately after their transaction.
- No unbounded arrays or materialized journey lifecycle are introduced.
- Very large histories require cursor/window exploration; a future projection is
  allowed only behind the same application contract and must be rebuildable.
