# PHINS Detector Dataset Governance

## Identity and ownership boundary

PHINS uses its own identifiers for the corpus it compiles and curates:

- owner namespace: `phins-group`
- founder/steward identifier: `duyhuynh`
- current plate corpus: `phins-vn-plate-corpus-v2`
- canonical sample prefix: `phins-vnplate-`
- canonical group prefix: `phins-group:plate-sequence:`

These identifiers establish ownership of the compilation structure, curation
logic, manifests, quality metadata, and derived model lifecycle. They do not
transfer copyright in third-party images to PHINS. Original author, URL,
revision, license, and source IDs remain in `ATTRIBUTION.csv` and
`PROVENANCE.jsonl`.

## Data layers

```text
External archive / reviewed warehouse captures
                    ↓
       canonical PHINS corpus source
                    ↓
       immutable detector dataset export
                    ↓
      training → evaluation → candidate model
```

Current directories:

```text
datasets/
├── source/plate/
│   └── roboflow/                       # canonical bbox sources
├── corpora/plate/
│   ├── phins-vn-plate-corpus-v1/       # immutable first compilation
│   └── phins-vn-plate-corpus-v2/       # current multi-source compilation
├── corpora/plate-auxiliary/
│   └── roboflow-yolov8-vit-folder-v2/  # classification; no bbox
└── detectors/plate/
    ├── phins-vn-plate-detection-v1/    # immutable first COCO export
    └── phins-vn-plate-detection-v2/    # current grouped COCO export
```

The external ZIP is not copied into the repository. Its SHA-256 is pinned in
the corpus manifest. Dataset directories remain ignored by Git and should be
stored only in approved private storage.

## Vietnam plate archive decision

The imported archive is Kaggle dataset `3543299`, version `6174984`, titled
“Vietnam License Plate Segment Datasets”. Kaggle identifies Duy Diệu Nguyễn and
a collaborator as authors but declares the license `Unknown`. Therefore the
corpus and derived dataset are deliberately marked:

```text
acceptanceEligible=false
releaseEligible=false
distributionEligible=false
licenseStatus=REVIEW_REQUIRED_UNKNOWN_SOURCE_LICENSE
```

Do not upload, redistribute, use for production model release, or treat it as
PHINS-owned source data until written permission or a clear commercial license
is recorded. Private research experiments may still require legal approval.

## Roboflow sources added in v2

All four downloaded archives declare `CC BY 4.0` in their embedded Roboflow
dataset cards. PHINS pins each archive SHA-256 and retains author, URL, version,
raw class, original path, and attribution evidence.

| Canonical source | Task | Raw images | Corpus use |
|---|---:|---:|---|
| `roboflow-traffic-violation-v3` | detection | 18,560 | plate detector |
| `roboflow-vietnamese-license-plate-v1` | detection | 8,357 | plate detector |
| `roboflow-license-plate-detection-v1` | detection | 2,555 | plate detector |
| `roboflow-yolov8-vit-folder-v2` | classification | 7,953 | auxiliary only |

The folder-classification source distinguishes motorcycle-plate context from
long car-plate context but contains no bounding boxes. It is deliberately kept
outside the detector corpus. Treating those images as detector negatives would
teach the model that visible plates are background.

The sources share 7,904 original lineages and include augmented variants.
Canonical group IDs are derived from the original pre-Roboflow filename, while
sample and image IDs are content-addressed PHINS identifiers. This keeps every
known variant of one source image in one split. Exact duplicate source bytes are
merged before composition.

Roboflow's CC BY declaration is preserved, but v2 remains non-releaseable
because it still includes the Kaggle source with an unknown license. A declared
license also does not replace provenance review for images originally collected
from elsewhere.

## Curation rules

- Source classes `BSD` and `BSV` map to one canonical detector class,
  `license_plate`; layout remains `ONE_LINE` or `TWO_LINE` metadata.
- Polygon annotations are preserved and exported as COCO segmentation while a
  canonical bounding box is derived for detector training.
- Exact duplicate images are merged by SHA-256. Overlapping annotation variants
  use a median polygon consensus rather than creating duplicate targets.
- Canonical image paths and sample IDs are content-addressed and contain no
  external author filename.
- Original train/validation labels are not trusted because every source
  sequence appears in both. Entire sequences are assigned to one split to
  prevent adjacent-frame and duplicate leakage.
- Roboflow augmentation variants are grouped by source lineage. Original source
  splits are recorded as metadata instead of being trusted blindly.
- Empty YOLO labels are retained as explicit negative samples only when the
  source provides a detector annotation contract.
- Brightness, contrast, sharpness, and estimated day/night attributes are
  retained for slice evaluation.
- Coordinate adjustments and rejected annotations are counted in the manifest;
  rejected records are written to `REJECTS.jsonl`.

## Reproducible commands

```bash
python run_model_training.py --config configs/model-training.yaml \
  ingest-roboflow-plate-archives \
  "/absolute/path/YOLOv8 and Vision transformer.v2i.folder.zip" \
  "/absolute/path/traffic_violation.v3i.yolov11.zip" \
  "/absolute/path/vietnamese license plate.v1i.yolov11.zip" \
  "/absolute/path/License Plate Detection.v1i.yolov11.zip"

python run_model_training.py --config configs/model-training.yaml \
  ingest-plate-corpus /absolute/path/to/archive.zip

python run_model_training.py --config configs/model-training.yaml \
  verify-corpus datasets/corpora/plate/phins-vn-plate-corpus-v2

python run_model_training.py --config configs/model-training.yaml \
  build-dataset --role plate --export-id phins-vn-plate-detection-v2

python run_model_training.py --config configs/model-training.yaml \
  verify-dataset datasets/detectors/plate/phins-vn-plate-detection-v2
```

Never mutate an existing version after build. A license decision, corrected
annotation, new source, or split-policy change produces a new corpus and
dataset version.

## Reviewed first-party versions and private Hub processing

The detector review workflow creates a new immutable first-party source instead
of editing its parent:

```text
phins-vn-plate-production-source-v1
  -> human decision revisions
  -> Promote Source
phins-vn-plate-production-source-v2
  -> source-bound immutable COCO export
  -> verified private Hugging Face commit
```

The `/datasets` catalog discovers only `FIRST_PARTY_DETECTOR_SOURCE` manifests
under the configured first-party root. A Hub sync is refused when reviews are
pending, the source is not release-eligible, the export does not bind to the
exact source-manifest SHA-256, credentials are absent, or the destination is
not private. The normal uploader continues to reject
`distributionEligible=false`.

Plate images are classified as `RESTRICTED_VEHICLE_IDENTIFIER`. An exceptional
private sync therefore requires all of the following at the same time:

1. the source asserts `USER_CONFIRMED_FIRST_PARTY_COLLECTION` and the expected
   proprietary first-party license status;
2. the source is release-eligible and has no pending review items;
3. `VIP_DATASET_REGISTRY__RESTRICTED_PRIVATE_SYNC_ENABLED=true` is explicitly
   enabled by the operator running the API;
4. an authenticated `ADMIN` confirms the restricted transfer for that request;
5. the Hugging Face API reports the target dataset repository as private before
   upload.

This is an external-processing exception, not a redistribution grant. The
manifest stays `distributionEligible=false`, the repository must remain
private, organizational access/retention approval still applies, and the data
must not be published or relicensed. It does not apply to the Kaggle/Roboflow
corpora described above; unresolved third-party source rights remain blocked.

Each request produces append-only audit evidence and a durable job record with
actor, source/export manifest hashes, destination, requested revision, status,
and the resulting Hub commit SHA. Tokens are runtime secrets and are never
stored in manifests, job records, audit metadata, or browser state.

## Path to a PHINS-owned production dataset

1. Obtain written rights for every third-party source or remove that source.
2. Collect warehouse/camera images under PHINS-approved consent and retention
   policy.
3. Assign track/passage group IDs before splitting.
4. Keep a site/camera/time-disjoint acceptance set that is never used for
   training or tuning.
5. Review low-light, blur, small-plate, one-line, two-line, motorcycle, and
   oblique-angle slices separately.
6. Build a new immutable corpus version with `releaseEligible=true` only after
   legal and data-governance approval.
