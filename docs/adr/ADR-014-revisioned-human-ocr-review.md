# ADR-014: Preserve AI prediction and revision human OCR review

- Status: Accepted
- Date: 2026-08-09

## Context

Low-confidence OCR needs operator correction, but overwriting
`plate.normalized` would destroy model-evaluation evidence. Multiple operators
can also open the same event concurrently. Training feedback must be durable and
idempotent without embedding images or an unbounded correction history in the
event.

## Decision

A reviewed event uses schema v2 with three explicit concepts:

- `plate.prediction` retains raw text, normalized text, confidence,
  observations, and character corrections from AI;
- `plate.review` stores the current reviewed value, reviewer identity, UTC time,
  note, and optimistic revision;
- `plate.final` is the effective indexed value used by event/vehicle search.

The repository applies the review with an atomic expected-revision predicate.
Identical retries are accepted idempotently; stale or different submissions
return conflict. Successful confirmation/correction is actor-audited. If a plate
crop object exists, a deterministic dataset sample ID derived from event ID and
review revision makes feedback creation replay-safe. `dataset_samples` stores
only an object key and bounded metadata.

Historical v1 documents are not bulk rewritten. Readers/search support an
indexed legacy fallback, and review lazily promotes the affected event to v2.

## Consequences

- Model quality can be measured against retained predictions.
- Concurrent operators cannot silently overwrite each other.
- Repeated API delivery does not duplicate training samples.
- Event documents stay bounded; audit history and samples live in separate
  collections.
- A standalone MongoDB deployment cannot atomically commit event amendment,
  dataset sample, and audit across collections. Retry repairs a missing
  deterministic sample, while strict all-or-nothing behavior requires a future
  replica-set transaction or transactional outbox.
- The current flow cannot assign a plate to an event that has no OCR prediction,
  and it does not automatically replay historical rules/actions after review.
