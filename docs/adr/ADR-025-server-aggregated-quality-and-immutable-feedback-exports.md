# ADR-025: Server-aggregated quality and immutable feedback exports

- Status: accepted
- Date: 2026-08-10

## Context

Quality reporting can accidentally scan all canonical events in an API process,
and random frame-level train/test splitting leaks fixed-camera appearance into
evaluation. Concurrent exporters can also duplicate or overwrite labeled data.

## Decision

Aggregate production quality inside MongoDB over an explicitly bounded UTC time
window and group by the model metadata stored on each event. Export human-review
feedback through leased dataset-sample states. Install one immutable directory
per stable export ID, checksum every artifact, and assign splits by a stable hash
of camera ID plus a versioned seed.

## Consequences

- Dashboard cost is proportional to the selected indexed time window, not API
  memory, while embedded development reports disclose capped scans.
- All observations from one camera stay in one split, reducing view leakage but
  potentially producing imbalanced small datasets.
- Retry verifies and reconciles an existing artifact; it cannot silently replace
  it.
- Human-review feedback remains selection-biased, so model promotion also needs
  a representative immutable holdout and explicit release gates.
