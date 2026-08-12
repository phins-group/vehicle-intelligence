# Vehicle and Plate Detector Training

## Boundary

Training is an offline model-lifecycle concern. The camera, tracking, OCR,
event, API, and dashboard paths do not import it:

```text
Reviewed source images + annotations
              |
              v
  immutable grouped COCO dataset
              |
              v
 PaddleDetection/PicoDet training
              |
              v
          ONNX export
              |
              v
 canonical Detector provider prediction
              |
              v
 metrics + configured release gates
              |
              v
 checksum model candidate / private registry
```

Modules have one responsibility:

| Module | Responsibility |
|---|---|
| `training.domain` | Canonical source annotations, roles, splits, result contracts |
| `training.config` | Strict vehicle/plate training, gate, Paddle, and Hub config |
| `training.bootstrap` | Pinned external sample adapters, attribution, safe source writer |
| `training.roboflow` | Pinned YOLO/folder archive import, class mapping, lineage grouping |
| `training.dataset` | Atomic COCO build, group split, byte verification |
| `training.inference` | Canonical provider to checksum-traced COCO predictions |
| `training.evaluation` | AP, precision/recall, group/slice/full-box metrics and gates |
| `training.paddledetection` | Isolated official train/export subprocesses |
| `training.artifacts` | Gate-passed ONNX candidate and model card packaging |
| `training.huggingface` | Private Hub upload and dataset/output-mounted Jobs |
| `training.cli` | Explicit operator composition root |

The checked-in configuration is
[`configs/model-training.yaml`](../configs/model-training.yaml). It contains
planning thresholds, not achieved performance claims.

## Training policy

- Vehicle and plate are independent models and artifacts.
- Vehicle classes are canonical `car`, `motorcycle`, `bus`, and `truck`.
- The plate detector has one class, `license_plate`. Layout is an evaluation attribute,
  not a detector class.
- A motorcycle plate annotation must enclose both rows.
- Use human-reviewed labels. A model suggestion is not ground truth until a
  reviewer accepts it.
- Start from a separately reviewed permissive checkpoint; do not put signed URLs
  or credentials in config.
- Framework, base checkpoint, source dataset, and company data rights must all be
  approved. Packaging deliberately records `licenseStatus=REVIEW_REQUIRED`.

## External bootstrap samples

The project can acquire small, reproducible samples without downloading an
entire public corpus:

```bash
python run_model_training.py --config configs/model-training.yaml \
  bootstrap-samples --role vehicle --samples-per-class 20

python run_model_training.py --config configs/model-training.yaml \
  bootstrap-samples --role plate --samples-per-class 20
```

Vehicle samples come from the official Open Images validation data and are
filtered to `Car`, `Motorcycle`, `Bus`, and `Truck`. Open Images states that its
annotations are CC BY 4.0 and images are listed as CC BY 2.0, while explicitly
requiring users to verify individual image licenses. The importer stores each
image's author, landing page, and license in `ATTRIBUTION.csv`.

Plate samples come from the pinned revision
`b76dbba86154c33fa370bc3087fbc7c766845a66` of
[`justjuu/license-plate-detection`](https://huggingface.co/datasets/justjuu/license-plate-detection),
whose card declares CC BY 4.0, one `license_plate` class, COCO `x, y, width,
height` boxes, and train/validation/test splits. It is a generic bootstrap
source, not a Vietnam-specific acceptance set.

Both importers write canonical `annotations.jsonl`, `source-manifest.json`,
`ATTRIBUTION.csv`, and checksummed images below the configured source directory.
They refuse to overwrite an existing source and can be verified independently:

```bash
python run_model_training.py verify-source datasets/source/vehicle
python run_model_training.py verify-source datasets/source/plate
```

Every imported sample has `bootstrapOnly=true` and
`acceptanceEligible=false`. Dataset build/evaluation may be used for smoke
testing, but candidate packaging rejects these datasets as release evidence.
Use reviewed warehouse-camera captures for validation, test, and production
acceptance.

For the PHINS founder namespace, Vietnam polygon archive ingestion, deduplication,
sequence-disjoint split policy, and the distinction between compilation identity
and third-party image ownership, see
[PHINS dataset governance](PHINS_DATASET_GOVERNANCE.md).

## Pinned Roboflow plate sources

Import the registered archives before composing plate corpus v2:

```bash
python run_model_training.py --config configs/model-training.yaml \
  ingest-roboflow-plate-archives \
  "/path/YOLOv8 and Vision transformer.v2i.folder.zip" \
  "/path/traffic_violation.v3i.yolov11.zip" \
  "/path/vietnamese license plate.v1i.yolov11.zip" \
  "/path/License Plate Detection.v1i.yolov11.zip"
```

Archives are identified by pinned SHA-256 rather than their filenames.
Detection classes such as raw `0` and `plate` are mapped at the source adapter
boundary to canonical `license_plate`. Roboflow augmentation variants with the
same original filename share a PHINS group ID, preventing split leakage.

The folder-classification archive is stored below
`datasets/corpora/plate-auxiliary` and is never passed to the detector builder
because it has no bounding boxes.

## Source dataset contract

### First-party production plate source

For images confirmed by the operator as first-party collection, build an
immutable production source by exact image identity:

```bash
python run_model_training.py --config configs/model-training.yaml \
  ingest-first-party-plate-images samples/images/plate \
  --source-id phins-vn-plate-production-source-v1 \
  --output datasets/source/plate-first-party/phins-vn-plate-production-source-v1 \
  --label-reference datasets/corpora/plate/phins-vn-plate-corpus-v2 \
  --auto-reference samples/extract/plate
```

The importer never joins annotations by filename or directory order. An image
enters canonical `annotations.jsonl` only when its exact SHA-256 matches a
canonical labeled image. Exact duplicates are collapsed while preserving a
hashed audit in `DUPLICATES.jsonl`. Images found only in the model-suggestion
set, and images with no verified annotation, are copied to `review/images` and
listed in `REVIEW_QUEUE.jsonl`; they cannot enter training until reviewed.

The source contract records owner namespace `phins-group`, steward
`duyhuynh`, `FIRST_PARTY_USER_COLLECTED`, proprietary first-party rights, and
restricted vehicle-identifier privacy. It permits internal production model
release but deliberately keeps dataset redistribution disabled.

Verify independently and then build the grouped COCO export:

```bash
python run_model_training.py verify-first-party-source \
  datasets/source/plate-first-party/phins-vn-plate-production-source-v1

python run_model_training.py --config configs/model-training.yaml \
  build-dataset --role plate \
  --export-id phins-vn-plate-production-v1
```

### Human review UI for detector labels

Run FastAPI and Angular from the repository root in separate terminals:

```bash
VIP_REALTIME__ENABLED=false VIP_LIVE_MONITOR__ENABLED=false vehicle-api

cd apps/web
npm run start
```

Open `http://localhost:4200/dataset-review`. The page is separate from
`/ocr-review`: OCR review corrects plate text on vehicle events, while detector
review draws `license_plate` bounding boxes on offline training images.

The backend discovers immutable first-party sources under
`dataset_review.sources_directory`. For the current production candidate this
is:

```text
datasets/source/plate-first-party/
  phins-vn-plate-production-source-v1/
    REVIEW_QUEUE.jsonl
    review/images/...
```

Review decisions never modify those files. Every submit creates a new,
no-overwrite revision under:

```text
datasets/reviews/detector/
  <source-id>/
    decisions/
      <review-id>/
        00000001.json
        00000002.json
```

The UI supports four explicit outcomes:

| Action | Result | Training meaning |
|---|---|---|
| Đúng như đề xuất | `APPROVED` | Preserve the original model bbox after human confirmation |
| Lưu bbox đã sửa | `CORRECTED` | Use one or more human-edited plate boxes |
| Không có biển số | `NEGATIVE` | Admit the image as a verified hard negative |
| Loại ảnh | `REJECTED` | Exclude an unusable/privacy-invalid image; a note is mandatory |

Coordinates are always source-image pixels. The backend repeats bounds and
action validation, binds every revision to the source manifest, queue checksum,
image SHA-256 and authenticated actor, and rejects stale `expectedRevision`
writes with HTTP `409`. The original suggestion and every human revision remain
available in history.

Recommended first pass for the current queue:

1. Select reason `Model đề xuất — cần xác nhận` to process the 960 auto-label
   samples.
2. Approve only when every visible plate is covered tightly.
3. Redraw incomplete/wrong boxes and save as `CORRECTED`.
4. Mark a genuine no-plate scene `NEGATIVE`; do not use `REJECTED` for useful
   hard negatives.
5. Review `Nhãn model xung đột`, then manually label the remaining unlabeled
   images.

Only `ADMIN` can start promotion. Promotion is an asynchronous, restart-aware
job that copies the verified parent samples and reviewed decisions into a new
immutable source ID (for example `phins-vn-plate-production-source-v2`). Pending
items remain pending in the new queue, rejected images are excluded, and
`REVIEW_DECISIONS.jsonl` preserves the promotion evidence. The parent `v1`
source is never edited. The job stores a checksummed map of the exact decision
revisions visible at start, so later operator edits cannot silently change an
already queued promotion. After the job reaches `COMPLETED`, point
`configs/model-training.yaml` at the new source for CLI training, or select that
source on `/datasets` and let the private-sync job build the export. Always use
a new COCO export ID; never reuse an ID belonging to another source manifest.

The current review queue lacks reliable sequence/camera metadata for all
unlabeled images. Newly promoted samples therefore use an image-identity group
and explicitly record `groupingBasis=SOURCE_IMAGE_SHA256_NO_SEQUENCE_METADATA`.
Do not use this reviewed queue as the untouched warehouse acceptance holdout;
keep the separately collected gold holdout grouped by real passage/camera.

### Generate proposals for unlabeled review images

Do not mark detector output as a human decision. The offline suggestion command
writes checksum-bound evidence below `datasets/reviews/detector` and leaves the
immutable source queue plus every existing human revision unchanged. Restart the
API after an offline run so the review repository loads the new overlays:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python run_model_training.py \
  --config configs/model-training.yaml \
  suggest-review-labels \
  datasets/source/plate-first-party/phins-vn-plate-production-source-v1 \
  --plate-model models/review/license-plate-finetune-v1s.pt \
  --model-name review-yolo11s-license-plate \
  --model-version hf-251a30d \
  --device mps \
  --image-size 1280 \
  --confidence 0.70 \
  --batch-size 4
```

Use `--device cpu` on machines without Apple Metal, and omit the MPS fallback
environment variable. The command skips source items that already have a model
proposal and items with a human decision. Every generated bbox remains
`PENDING_REVIEW`; the operator must still approve, correct, mark negative, or
reject it. If a partially reviewed source is promoted, pending model suggestion
overlays are carried into the new immutable source queue.

Relevant API endpoints are:

```text
GET  /api/detector-review/sources
GET  /api/detector-review/items
GET  /api/detector-review/sources/{sourceId}/items/{reviewId}
GET  /api/detector-review/sources/{sourceId}/items/{reviewId}/image
PUT  /api/detector-review/sources/{sourceId}/items/{reviewId}
GET  /api/detector-review/sources/{sourceId}/items/{reviewId}/history
POST /api/detector-review/sources/{sourceId}/promotions
GET  /api/detector-review/promotions/{jobId}
```

### Dataset catalog and private Hub sync after promotion

Open `http://localhost:4200/datasets` after promotion. The screen lists every
immutable first-party source version, its parent, source-manifest SHA-256,
review count, matching COCO export, and latest Hugging Face job. Selecting a
promoted source and choosing **Build, verify và sync private Hub** runs this
server-side sequence:

Use the **Xem mẫu Dataset** section on the same page to inspect what was
actually promoted before exporting. It reads 12 samples per page, supports
`ALL`, `POSITIVE`, and `NEGATIVE` plus `DAY`, `NIGHT`, and `UNKNOWN` filters,
and renders every canonical `license_plate` bbox over its image. The detail
panel exposes split, camera/group metadata, review state, SHA-256, dimensions,
and exact pixel coordinates. The promoted source remains immutable; use the
detector review and a new promotion version if a label must change.

```text
selected immutable source
  -> reject pending/unreleaseable source
  -> build or reuse a source-bound COCO export
  -> verify every export checksum and source-manifest binding
  -> create/verify the configured Hugging Face repository as private
  -> upload and retain the returned commit SHA/URL in durable job evidence
```

The browser never receives `HF_TOKEN`, source file paths, or image bytes for
this operation. `OPERATOR` may inspect the catalog; only `ADMIN` can start a
sync. Audit persistence succeeds before the background uploader is dispatched.
Identical completed requests are idempotent, and an active job prevents a
second export/upload for the same source.

Host-native API can use the credential already stored by `hf auth login`. A
container cannot see that host credential cache, so supply `HF_TOKEN` to Docker
Compose. For restricted first-party plate data, a second server-side switch is
deliberately off by default:

```dotenv
# .env — keep this file private and never commit it
HF_TOKEN=hf_your_write_scoped_token
VIP_DATASET_REGISTRY__RESTRICTED_PRIVATE_SYNC_ENABLED=true
```

```bash
docker compose up -d --build api web
```

Then open `/datasets`, select the promoted `v2` source, keep the suggested new
export ID, tick the restricted-transfer confirmation, and start the job. The
remote repository configured in `configs/model-training.yaml` must belong to
the authenticated namespace and is checked as private immediately before any
upload. A least-privilege fine-grained token with write access only to that
private dataset repository is recommended.

The corresponding API is asynchronous:

```text
GET  /api/datasets
GET  /api/datasets/{sourceId}/samples?limit=12&cursor=...&kind=...&lighting=...
GET  /api/datasets/{sourceId}/samples/{imageSha256}/image
POST /api/datasets/{sourceId}/syncs
GET  /api/datasets/syncs/{jobId}
```

The sample cursor is bound to the source ID, source-manifest SHA-256 and active
filters. Metadata listing streams `annotations.jsonl` from the cursor offset
instead of loading the full corpus into memory. Before serving a preview, the
API verifies manifest membership, size, SHA-256, decodability, pixel policy and
bbox bounds; it never returns source filesystem paths.

`QUEUED`, `PREPARING_EXPORT`, and `UPLOADING` are active states; the terminal
states are `COMPLETED` and `FAILED`. Job JSON evidence is stored under
`datasets/registry/huggingface/jobs` and survives API/container restarts.

`distributionEligible=false` remains unchanged. The explicit switch and ADMIN
confirmation authorize only approved external processing in a verified private
repository for a release-eligible, user-confirmed first-party source. They do
not authorize public publication, sharing, relicensing, or syncing any
third-party corpus whose rights remain unresolved.

Docker Compose bind-mounts `./datasets` into the API container so source images,
review revisions and promoted versions survive container rebuilds.

### Extract review candidates from traffic videos

Use the offline extraction command to turn a directory of videos into a staged,
traceable review set for both detector roles:

```bash
python run_model_training.py --config configs/model-training.yaml \
  extract-video-samples /path/to/videos \
  --output samples/extract \
  --vehicle-model models/yolo11n.pt \
  --plate-model models/vietnam-plate.pt \
  --device cpu \
  --sample-interval-seconds 1
```

The output separates the actual detector inputs from crop previews:

```text
samples/extract/
├── vehicle/
│   ├── images/                  # traffic frames; vehicle detector input
│   ├── crops/<class>/           # car/motorcycle/bus/truck review previews
│   └── annotations.auto.jsonl   # suggested vehicle boxes/classes
├── plate/
│   ├── images/                  # vehicle contexts; plate detector input
│   ├── crops/                   # plate review previews
│   └── annotations.auto.jsonl   # suggested license_plate boxes
└── manifest.json                # video hashes, settings, counts, data policy
```

Frames from one source video share a group ID, so adjacent frames cannot leak
between train and evaluation splits. Full-resolution source frames are retained
for vehicle/plate crops while detector frames are bounded by
`--detector-frame-max-edge` for throughput. The source videos themselves are not
copied.

`annotations.auto.jsonl` is deliberately not named `annotations.jsonl`: YOLO and
plate-model output is an annotation suggestion, not ground truth. Review boxes,
vehicle classes, two-row plate coverage, missed objects, and false positives
before promoting reviewed records into a canonical source directory. If source
rights were not supplied, the extraction manifest remains
`licenseReviewStatus=REVIEW_REQUIRED`, `releaseEligible=false`, and
`distributionEligible=false`.

Each source directory contains images and newline-delimited `annotations.jsonl`:

```text
datasets/source/vehicle/
├── annotations.jsonl
└── images/
    ├── gate01-track100-frame10.jpg
    └── ...

datasets/source/plate/
├── annotations.jsonl
└── images/
    ├── gate01-track100-vehicle.jpg
    └── ...
```

Each JSON line describes one independently reviewable image:

```json
{
  "sampleId": "gate01-track100-frame10",
  "imagePath": "images/gate01-track100-frame10.jpg",
  "groupId": "vehicle-passage-100",
  "cameraId": "gate01",
  "capturedAt": "2026-08-10T08:30:00+07:00",
  "split": null,
  "attributes": {
    "lighting": "NIGHT",
    "weather": "DRY"
  },
  "annotations": [
    {
      "className": "car",
      "bbox": {
        "x": 120,
        "y": 240,
        "width": 800,
        "height": 430
      },
      "attributes": {
        "occluded": false
      }
    }
  ]
}
```

Bounding boxes are pixel `x, y, width, height`. `imagePath` must remain below the
source directory. Empty `annotations` is a valid reviewed negative image.

`groupId` is the leakage boundary. Use a stable vehicle passage/identity when
known; otherwise use the track. Adjacent frames and cross-camera views of the
same known vehicle must share a group. An optional explicit `split` can be
`train`, `validation`, or `test`; conflicting declarations in one group fail the
build. Unassigned groups use a stable seeded hash.

Recommended attributes for slice evaluation are:

```text
lighting: DAY | NIGHT
layout: SINGLE_LINE | TWO_LINE
readable: true | false
occluded: true | false
glare: true | false
```

CVAT Community can be used for labeling, but its export must be transformed into
this reviewed contract with the platform-owned group/camera/time metadata. Do
not infer those fields from filenames in a release dataset.

## Build and verify immutable datasets

```bash
python run_model_training.py --config configs/model-training.yaml \
  build-dataset --role vehicle --export-id warehouse-vehicle-v1

python run_model_training.py --config configs/model-training.yaml \
  build-dataset --role plate --export-id warehouse-plate-v1

python run_model_training.py verify-dataset \
  datasets/detectors/vehicle/warehouse-vehicle-v1
```

The output contains copied source bytes, COCO `train`, `validation`, and `test`
documents, and a SHA-256 manifest. The builder rejects unsafe paths, invalid
images/boxes, duplicate sample IDs, cross-group duplicate bytes, empty required
splits, and group leakage. Existing exports are never overwritten.

With only four cameras, use an operational time/track holdout plus explicit
unseen-camera groups. Do not randomize individual frames.

## PaddleDetection setup

The backend intentionally does not vendor or silently download a training
framework. Review and pin one official PaddleDetection revision, install its
documented Paddle/Paddle2ONNX versions in a dedicated environment or image, and
place the checkout at the configured `third_party/PaddleDetection` path. The
default base config is `configs/picodet/picodet_m_416_coco_lcnet.yml` relative to
that checkout.

Inspect the exact no-shell command before consuming GPU:

```bash
python run_model_training.py --config configs/model-training.yaml \
  train --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --run-id vehicle-picodet-m-v1 --dry-run
```

Run training:

```bash
python run_model_training.py --config configs/model-training.yaml \
  train --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --run-id vehicle-picodet-m-v1

python run_model_training.py --config configs/model-training.yaml \
  train --role plate \
  datasets/detectors/plate/warehouse-plate-v1 \
  --run-id plate-picodet-m-v1
```

Every run has a log and `training-run.json` containing dataset/config hashes,
command, timing, exit status, and PaddleDetection Git revision when available.
Failure and timeout still leave evidence; they never create a successful model
candidate.

Export selected weights through the official Paddle inference exporter and
Paddle2ONNX checker:

```bash
python run_model_training.py --config configs/model-training.yaml \
  export-onnx --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --weights output/training/vehicle/vehicle-picodet-m-v1/best_model.pdparams \
  --output output/models/vehicle-picodet-m-v1.onnx
```

Paddle/PaddleDetection/Paddle2ONNX compatibility is version-sensitive. A failed
export is a blocker; do not rename a Paddle checkpoint to `.onnx` or bypass the
checker.

## Create canonical prediction evidence

Evaluation must use the same provider boundary as production. The command below
loads the ONNX candidate through `PicoDetDetector`, returns canonical detections,
and writes COCO predictions plus a checksum sidecar:

```bash
python run_model_training.py --config configs/model-training.yaml \
  predict --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --split test \
  --runtime-config configs/default.yaml \
  --provider picodet \
  --model output/models/vehicle-picodet-m-v1.onnx \
  --model-name warehouse-vehicle \
  --model-version v1 \
  --image-size 416 \
  --output output/evaluation/vehicle-v1-predictions.json
```

For plate, switch the role, dataset, and model. For a custom vehicle model the
class order defaults to the training target classes; use `--model-classes` only
when the exported order is different and documented.

## Evaluate and package

```bash
python run_model_training.py --config configs/model-training.yaml \
  evaluate --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --split test \
  --predictions output/evaluation/vehicle-v1-predictions.json \
  --output output/evaluation/vehicle-v1.json
```

Exit code `4` means measured output failed one or more release gates. Reports
include AP50, mAP50-95, operational precision/recall, per-class, per-camera,
day/night, group recall, and full-ground-truth-box coverage. Missing NIGHT or a
critical class fails a configured gate instead of being silently ignored.

Only a checksum-traced test report that passes all gates can become a candidate:

```bash
python run_model_training.py --config configs/model-training.yaml \
  package --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1 \
  --onnx output/models/vehicle-picodet-m-v1.onnx \
  --evaluation output/evaluation/vehicle-v1.json \
  --training-run output/training/vehicle/vehicle-picodet-m-v1/training-run.json \
  --model-name warehouse-vehicle \
  --model-version v1 \
  --output output/model-candidates
```

Packaging validates ONNX, freezes the dataset/evaluation/training evidence,
generates a model card, and hashes every file. It does not auto-promote the model
into the camera config. Shadow comparison and human approval remain separate.

## Private Hugging Face integration

Install optional clients and authenticate without putting tokens in YAML:

```bash
python -m pip install -e '.[training]'
hf auth login
```

The checked-in configuration targets the private PHINS repositories
`phins-group/vehicle-dataset`, `phins-group/plate-dataset`,
`phins-group/vehicle-detector`, and `phins-group/plate-detector`. Hub upload is
enabled independently from remote training. Dataset/model upload verifies the
local manifest and confirms the remote repo is actually private before any bytes
are sent:

```bash
python run_model_training.py --config configs/model-training.yaml \
  hf-upload-dataset --role vehicle \
  datasets/detectors/vehicle/warehouse-vehicle-v1
```

Every uploadable dataset export contains a manifest-hashed `README.md`,
`ATTRIBUTION.csv`, and, for acceptance-ineligible data, `BOOTSTRAP_ONLY.md`.
The Hub card deliberately declares `license: other`; the source-code
Apache-2.0 license does not relicense third-party training images. The uploader
rejects older exports that lack these verified provenance files, so rebuild an
old export under a new immutable export ID before uploading it.

A Job receives the dataset read-only at `/data` and a bucket read-write at
`/output`. Jobs remain disabled until `huggingface.jobs_enabled=true`, a pinned
`job_image`, and a persistent private `job_output_bucket` are configured. The
custom image must contain the project, pinned PaddleDetection checkout,
compatible Paddle/Paddle2ONNX packages, and an HF-specific training config whose
output path is below `/output`:

```bash
export HF_TOKEN='...'
python run_model_training.py --config configs/model-training.yaml \
  hf-submit-job --role vehicle \
  --secret-from-local HF_TOKEN \
  --name vehicle-picodet-v1 \
  -- vehicle-model-training --config /workspace/configs/model-training.hf.yaml \
     train --role vehicle /data --run-id vehicle-picodet-v1
```

Secrets are read by name from the local environment and sent through the Jobs
secret channel; they are not accepted as command arguments or written to model
manifests. Hugging Face compute is optional and has independent billing. It does
not change framework, checkpoint, or dataset licenses.

Do not upload continuous warehouse video. If company policy forbids plate data
leaving the site, train locally and upload only the approved candidate package,
or keep the entire registry on premises.

## Current honest limitations

- No real warehouse detector dataset is committed.
- No trained PicoDet checkpoint or accuracy result is claimed.
- The operator must pin a PaddleDetection/Paddle/Paddle2ONNX compatibility set.
- CVAT-to-canonical metadata transformation remains an annotation operations
  step because CVAT alone does not know platform track/identity split groups.
- Model promotion remains manual and requires site shadow/soak evidence.
