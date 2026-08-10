# Vehicle Re-identification

## Candidate-to-decision flow

```text
current fingerprint
  -> enabled inbound topology edges
  -> indexed travel-time candidates
  -> plate / embedding / type / color / time signals
  -> versioned weighted score
  -> MATCH | REVIEW | REJECT recommendation
  -> explicit reviewed merge or split
```

The scoring service never mutates an identity. Optional signals are omitted when
evidence is unavailable; remaining configured weights are renormalized. Visual
cosine is used only when both references have identical model
name/version/dimension/hash. Scores and thresholds are configuration, not domain
constants.

`MATCH` does not mean automatic merge. An OPERATOR or ADMIN submits a stable
`reviewId`, source/target revisions, a reason, and optionally the exact scored
fingerprint pair. The server recomputes feasibility and score for supplied
evidence. Merge moves all source fingerprints/events into the target, marks the
source identity `MERGED`, recalculates the bounded target aggregate, stores the
review, and audits the actor. Split selects a strict subset of fingerprints,
creates a deterministic new identity, recalculates both aggregates, relinks their
events, stores the review, and audits the actor.

All Mongo mutations share one transaction. Retrying an identical `reviewId`
returns the stored result without another mutation or audit. A stale revision,
changed ownership, attempt to split every observation, or reused review ID with
different intent returns `409`.

## API

```text
GET  /api/vehicle-fingerprints/{id}/reid-candidates
POST /api/vehicle-identities/merge
POST /api/vehicle-identities/split
GET  /api/vehicle-identity-reviews/{reviewId}
```

Review operations are bounded to 1,000 fingerprints. This keeps transaction
duration/document work predictable; exceptionally large identities require an
offline administrative workflow rather than an unbounded API transaction.
