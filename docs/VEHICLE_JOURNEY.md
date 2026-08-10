# Vehicle Journey and Timeline

## Purpose

A journey is a read model derived from canonical `vehicle_events` linked to one
reviewed logical `vehicleId`. It answers where and when a logical vehicle was
observed without creating another source of truth.

## API

```text
GET /api/vehicles/{vehicleId}
GET /api/vehicles/{vehicleId}/timeline?from=&to=&limit=
GET /api/vehicles/{vehicleId}/journey?from=&to=&limit=
```

The identity response includes the latest canonical event. Timeline items are
always chronological. Journey adds one segment between each consecutive pair of
observations. If a matching directed topology edge exists, the segment contains
its configured travel-time window and a `feasible` result. Missing topology is
reported as `feasible: null`; it is never treated as proof that travel was valid.

Reads are bounded by `identity.journey_event_limit`. A journey fetches at most
one extra event to set `truncated` explicitly and never embeds history into the
`vehicles` document.

## Operator console

`/vehicles/:vehicleId` renders the logical identity, plate aliases, latest
observation, chronological camera steps, elapsed travel time, and topology
violations. Event details remain canonical and are loaded on demand through the
existing event API and signed-media boundary.

## Consistency

Merge and split reviews update `vehicle_events.vehicleId` in the same transaction
as identities and fingerprints. Journey reads therefore reflect the reviewed
ownership after commit. The read model is intentionally not materialized; if
scale later requires projection, it must preserve event IDs and rebuild safely.
