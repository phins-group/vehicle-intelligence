# Phase 3 ReID and Identity Review Acceptance

Accepted on 2026-08-10.

## Implemented

- Versioned weighted ReID score across normalized plate edit similarity,
  compatible embedding cosine, vehicle type, color, and topology travel time.
- Configurable `MATCH`, `REVIEW`, and `REJECT` thresholds with missing-signal
  weight renormalization and strict embedding model compatibility.
- Read-only scored candidate API; scoring cannot mutate identity.
- OPERATOR/ADMIN merge and split review APIs with stable review IDs, reasons,
  expected revisions, optional server-revalidated fingerprint evidence, and audit.
- Transactional Mongo identity aggregate/fingerprint/event relinking plus immutable
  `identity_reviews`; identical retry is idempotent and conflicting reuse fails.
- Split is a reviewed reverse operation that moves a strict fingerprint subset to
  a deterministic new identity and recalculates both bounded aggregates.

## Tested

- Unit tests cover multi-signal scoring, vector contribution, non-mutation,
  verdicts, stale revisions, merge, split, event/fingerprint relinking, and retry.
- Real MongoDB replica-set integration verifies transactional merge/split,
  lifecycle status, event links, review persistence, and idempotent replay.
- API integration verifies score explanation, merge/split review, retrieval, and
  exactly one audit per non-idempotent mutation.
- Full real-service suite: **173 passed, 0 skipped**; Ruff passes.

## Known limitations

- No automatic merge is enabled; a human review remains mandatory.
- Brand/model and learned topology probability are not yet score signals.

## Next

Journey generation, logical vehicle detail/timeline, and Angular investigation UI.
