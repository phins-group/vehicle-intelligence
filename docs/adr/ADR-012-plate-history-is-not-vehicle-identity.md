# ADR-012: Plate History Is Not Vehicle Identity

- Status: Accepted
- Date: 2026-08-09

## Context

Operators need to investigate where and when a normalized plate appeared before
Phase 3 vehicle identity and ReID exist. The event store already has an indexed
`plate.normalized + occurredAt` access path. Calling every exact plate match one
vehicle would nevertheless be incorrect: OCR can be wrong, plates can be cloned
or reassigned, and some observations already have different or null logical
`vehicleId` values.

The original search endpoint returned one bounded list without a continuation
cursor, which made a UI timeline unable to distinguish the loaded window from
complete history.

## Decision

Treat `/api/vehicles/search` as an exact canonical plate-observation query. Use
the existing `VehicleEventRepository.list(EventQuery)` path so the endpoint gains
opaque cursor pagination while retaining its normalized query and event items.
Keep results newest-first at the API and let presentation layers order only their
loaded page set for chronological display.

In Angular, name the surface “Tra cứu biển số”, label counts and time boundaries
as loaded evidence, cap browser state at 500 events, and expose assigned/null
vehicle identity state without merging it. Support a shareable `plate` query
parameter and navigation from an event detail.

Do not add fuzzy search until an indexed derived field or replaceable search
adapter can bound candidates. Do not infer physical identity from plate matches.

## Consequences

- Investigations can page through exact plate history without a collection scan.
- Existing API consumers remain compatible because `query` and `items` are
  unchanged; `nextCursor` is additive.
- UI summaries cannot silently claim a total or first-seen value before all
  pages are loaded.
- A logical vehicle detail/journey still depends on Phase 3 identity, embedding,
  topology, time constraints, and human merge/split workflows.
- Fuzzy matching, signed evidence media, and identity confidence remain future
  milestones rather than being simulated in the browser.
