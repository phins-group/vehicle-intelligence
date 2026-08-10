# Model Quality and Retraining Feedback

## Purpose

The quality loop connects immutable production evidence to measured model
performance without placing training logic in the vision pipeline:

```text
VehicleEvent + AI model trace
        |                Human plate review
        |                         |
        v                         v
Mongo server aggregation    dataset_samples
        |                         |
        v                         v
/api/model-quality       leased dataset exporter
        |                         |
        v                         v
Angular dashboard       immutable camera-grouped export
                                  |
                                  v
                         offline evaluation gates
```

Training and deployment remain explicit offline operations. A corrected sample
never changes the original AI prediction, and an export never promotes a model
automatically.

## Quality report

`GET /api/model-quality?from=<ISO-8601>&to=<ISO-8601>` requires platform-read
permission. Both timestamps must contain a timezone; the default is the last 30
days and the configured maximum is 365 days. MongoDB performs the bounded time
match and grouping server-side. Local/JSONL development uses a capped scan and
sets `truncated=true` when that cap is reached.

Metrics use explicit denominators:

- `ocrSuccessRate = readablePlateCount / eventCount`;
- `unknownPlateRate = (NO_PLATE + UNREADABLE) / eventCount`;
- `humanCorrectionRate = correctedCount / reviewedCount`;
- `averagePlateConfidence` is calculated only over events with plate evidence.

The response contains totals, UTC daily points, per-OCR-model name/version/hash
slices, and feedback counts by export state/reason. Event model metadata is the
source of truth; missing historical metadata is reported as an unknown model,
not attributed to the current deployment.

## Dataset state machine

```text
READY ──claim──> EXPORTING ──verified artifact──> EXPORTED
                    |
                    +──sample/build error──> EXPORT_FAILED ──retry──> EXPORTING

stale EXPORTING lease ──reclaim──> EXPORTING
```

Claims are oldest-first, bounded, atomic per Mongo document, resumable by stable
`exportId`, and carry attempts/claim time. `READY`, `EXPORTING`, and
`EXPORT_FAILED` samples pin their source event and media against retention.
`EXPORTED` is terminal for that sample. An existing valid export is checksum
verified and only reconciles database state; it is never overwritten.

## Export command and artifact

```bash
vehicle-dataset-export \
  --config configs/default.yaml \
  --export-id ocr-20260810-v1 \
  --limit 500
```

Equivalent source-tree command:

```bash
python run_dataset_export.py --export-id ocr-20260810-v1
```

The output is installed atomically under the configured root:

```text
datasets/exports/ocr-20260810-v1/
├── images/
│   ├── train/
│   ├── validation/
│   └── test/
├── labels.jsonl
└── manifest.json
```

Every image and `labels.jsonl` entry has a SHA-256 and byte size in the manifest.
Input media reads are byte-bounded, decoded, dimension-checked, and normalized to
JPEG. Both the maximum encoded bytes and decoded pixel count are configurable;
relative paths are containment-checked. The manifest records model mix,
split counts, sample IDs, split seed/ratios, and creation time.

Train/validation/test assignment hashes `splitSeed + cameraId`. All samples from
one camera therefore remain in exactly one split, preventing near-identical
frames from the same fixed view leaking across evaluation boundaries. Changing
the split seed is a versioned data decision and produces a different dataset
lineage.

## Offline evaluation and release gates

```bash
python scripts/evaluate_ocr_dataset.py datasets/exports/ocr-20260810-v1 \
  --minimum-exact-accuracy 0.95 \
  --minimum-character-accuracy 0.99 \
  --maximum-ece 0.08 \
  --output output/benchmarks/ocr-20260810-v1-evaluation.json
```

Evaluation first verifies every manifest checksum, then reports exact plate
accuracy, micro character accuracy, expected calibration error, and breakdowns
by split/model. A failed threshold exits non-zero for CI/CD promotion gates.

Human-review samples are deliberately enriched for errors and are therefore
selection-biased. They are valuable regression cases but cannot be the only
release set. Production promotion still requires a separately curated,
representative and immutable holdout plus the detector/runtime and edge-capacity
gates documented in `MODEL_OPTIMIZATION.md` and `EDGE_DEPLOYMENT.md`.

## Operational safeguards

- No image bytes or embeddings are stored in MongoDB.
- Export failures use machine-readable codes and do not delete source media.
- No exporter path can escape its configured root and no valid export is
  overwritten.
- API reports are private/no-store and protected by RBAC.
- Model promotion is manual and requires an updated model hash, version,
  configuration version, edge manifest, and benchmark evidence.
