# Model Requirements

Model binaries are deliberately not committed. Put licensed, evaluated
checkpoints in this directory or pass an absolute path.

## Vehicle detector

The default `yolo11n.pt` is an Ultralytics COCO detector. The adapter filters to
`car`, `motorcycle`, `bus`, and `truck`. Production deployments should pin the
resolved model file, version, optional SHA-256, class mapping, and evaluation
report instead of relying on an implicit download.

PicoDet uses `provider: picodet` with a PaddleDetection-compatible ONNX export.
Set the exact ordered `model_classes`, input size, model version, and SHA-256.
The repository does not currently include an evaluated PicoDet artifact; unit
tests validate preprocessing/post-processing, while the real PicoDet smoke test
remains explicitly skipped until `models/picodet.onnx` is supplied.

## Vietnamese plate detector

`vision.plate_detection.model_path` is mandatory. It can be a
YOLO/Ultralytics-compatible detection or oriented-bounding-box checkpoint, or a
PicoDet-compatible ONNX export trained/evaluated on Vietnamese plates. Detection
coordinates must be relative to the vehicle crop. For a YOLO OBB result, the
adapter preserves four corners for perspective correction; the current PicoDet
adapter is axis-aligned and does not fabricate perspective geometry.

### Local demo checkpoint

The project does not endorse or automatically download a community checkpoint.
For a local integration demo only, the MIT-licensed
[`Koushim/yolov8-license-plate-detection`](https://huggingface.co/Koushim/yolov8-license-plate-detection)
model is an Ultralytics YOLOv8n detector with one `license_plate` class and a
640-pixel input. It is a generic international plate detector and has not been
evaluated on Vietnamese car/motorcycle plates, night scenes, two-line plates, or
the deployment cameras for this project.

Download the exact reviewed artifact revision instead of the mutable `main`
branch:

```bash
curl -L --fail \
  'https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/83c98fbe7412fe8b3950adb5637cfd08b0f04809/best.pt' \
  --output models/vietnam-plate.pt
```

Verify the artifact before loading it:

```bash
shasum -a 256 models/vietnam-plate.pt
```

Expected SHA-256:

```text
2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822
```

The expected size is `6,248,291` bytes. The source revision and LFS checksum are
recorded in the upstream
[`83c98fb` commit](https://huggingface.co/Koushim/yolov8-license-plate-detection/commit/83c98fbe7412fe8b3950adb5637cfd08b0f04809).
Treat every PyTorch checkpoint as executable model input: download only from a
source you trust and never replace the pinned artifact without reviewing and
recording its new hash.

Confirm that Ultralytics can load it and that its class map is correct:

```bash
python -c \
  "from ultralytics import YOLO; model=YOLO('models/vietnam-plate.pt'); print(model.names)"
```

Expected class map:

```text
{0: 'license_plate'}
```

Use the checkpoint without changing pipeline source code:

```yaml
vision:
  plate_detection:
    provider: yolo
    model_path: models/vietnam-plate.pt
    model_name: generic-license-plate-demo
    model_version: hf-83c98fb
    model_hash: 2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822
    confidence: 0.30
    iou: 0.45
    image_size: 640
    model_classes: [license_plate]
```

Or override only the artifact path for a local CLI run:

```bash
python run_camera.py \
  --camera laptop-webcam \
  --rtsp rtsp://127.0.0.1:8554/webcam \
  --vehicle-model models/yolo11n.pt \
  --plate-model models/vietnam-plate.pt \
  --storage local \
  --no-mongo
```

This checkpoint is suitable for wiring and smoke testing only. Before production,
fine-tune and evaluate a pinned model on a licensed Vietnamese dataset and the
actual camera distribution. Record the resulting model hash, class map, split,
metrics, and deployment acceptance evidence below.

### Annotation-assistant checkpoint

The offline detector-review queue may use the larger
[`morsetechlab/yolov11-license-plate-detection`](https://huggingface.co/morsetechlab/yolov11-license-plate-detection)
YOLO11s checkpoint to propose boxes that a human must review. It is pinned to
revision `251a30d7daedca065f56e04b0af04052c907c68f`:

```bash
mkdir -p models/review
hf download morsetechlab/yolov11-license-plate-detection \
  license-plate-finetune-v1s.pt \
  --revision 251a30d7daedca065f56e04b0af04052c907c68f \
  --local-dir models/review

shasum -a 256 models/review/license-plate-finetune-v1s.pt
```

Expected SHA-256:

```text
95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381
```

This upstream checkpoint is AGPL-3.0 and its model card reports contamination in
the source train/test split. It is therefore an annotation assistant only: it is
not the PHINS production model, does not establish production accuracy, and its
suggestions never become training labels without a recorded human decision.

Before accepting a checkpoint, record:

- license and source dataset;
- model/version/hash and class mapping;
- train/validation split without camera leakage;
- day/night, car/motorcycle, blur, angle, and province/series coverage;
- precision/recall and downstream OCR success at the configured threshold.

The repository does not download a community checkpoint automatically because
unknown licensing, label conventions, and quality would make a successful demo
look like a validated production model.

## Optimized artifacts

Use `scripts/export_detector_model.py` to create ONNX or TensorRT artifacts from
a pinned real checkpoint. Keep the emitted model/version/source hash/artifact
hash manifest with deployment configuration. The runtime verifies an optional
configured hash and refuses a requested accelerator that is unavailable. See
[`MODEL_OPTIMIZATION.md`](../docs/MODEL_OPTIMIZATION.md).

## OCR

Defaults are `PP-OCRv5_mobile_det` plus `PP-OCRv5_mobile_rec`. PaddleOCR model
files download into its configured cache on first use. Raw text remains separate
from Vietnamese normalization and temporal voting.
