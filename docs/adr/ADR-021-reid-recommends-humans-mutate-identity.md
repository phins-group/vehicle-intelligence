# ADR-021: ReID recommends; reviewed operations mutate identity

## Status

Accepted — 2026-08-10.

## Decision

ReID combines plate edit similarity, compatible visual embedding cosine,
vehicle type, color, and topology travel-time evidence using a versioned,
configurable weighted score. Missing optional signals cause weight
renormalization; incompatible embedding versions are never compared.

Even a `MATCH` verdict is a recommendation. Identity merge/split requires an
authenticated reviewer, reason, expected revisions, and stable review ID. The
Mongo adapter moves aggregate state, fingerprints, and canonical event links in
one replica-set transaction, then stores the immutable review. Retries with the
same request are idempotent; reuse with different intent is rejected.

## Consequences

- False positives cannot silently rewrite vehicle history.
- Every identity mutation is attributable and reversible through a reviewed split.
- Operations are bounded to 1,000 fingerprints to avoid unbounded transactions.
- Future auto-match may be enabled only behind a separately accepted policy.
