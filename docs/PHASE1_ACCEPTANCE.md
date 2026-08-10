# Phase 1 Acceptance Record

## Decision

The Phase 1 execution path passed an engineering acceptance run on 2026-08-09.
The run used real Ultralytics, ByteTrack, PaddleOCR, OpenCV, temporal voting,
direction estimation, JSONL persistence, and local media storage. It emitted one
logical event from one logical track.

This is not a production accuracy claim. A deployment checkpoint still requires
evaluation on representative natural video from each target camera.

## Test provenance

- Source image: [Vietnamese license plate for Cao Bang province](https://commons.wikimedia.org/wiki/File:Vietnamese_license_plate_for_Cao_B%E1%BA%B1ng_province.jpg),
  by Wikimedia Commons user Bún bòa, licensed CC BY-SA 4.0.
- Test transformation: the source image was scaled and panned into a temporary
  960x720, 6 FPS, 18-frame H.264 video. Neither source media nor derived media is
  committed to this repository.
- Vehicle model: Ultralytics `yolo11n.pt`, COCO vehicle classes, temporary local
  acceptance artifact.
- Plate model: `Koushim/yolov8-license-plate-detection`, class
  `license_plate`, temporary local acceptance artifact. SHA-256:
  `2d95861825bb4184404344c9cf809f40fd31dba785fe54e8ba5b9a3583789822`.
  Its model card labels it MIT, while its Ultralytics base has separate licensing;
  it is therefore not shipped or approved as the production checkpoint.
- OCR: `PP-OCRv5_mobile_det` and `PP-OCRv5_mobile_rec` via PaddleOCR.
- Execution device: CPU on the local Intel macOS development machine.

## End-to-end result

```json
{
  "eventType": "VEHICLE_ENTER",
  "direction": "ENTER",
  "status": "CONFIRMED",
  "plate": {
    "raw": "11A-01568",
    "normalized": "11A-015.68",
    "confidence": 0.9228224562548191,
    "observationCount": 18
  },
  "vehicle": {
    "type": "car",
    "confidence": 0.6712536613146464
  },
  "stats": {
    "frames": 18,
    "plateDetections": 18,
    "plateObservations": 18
  }
}
```

Exactly one JSONL document and three JPEG objects were written: full snapshot,
vehicle crop, and plate crop. The plate crop was visually checked against the
source and reads `11A-015.68`.

## CPU benchmark

| Component | Count | Mean | p50 | p95 |
|---|---:|---:|---:|---:|
| Vehicle inference | 18 | 187.691 ms | 107.589 ms | 484.069 ms |
| Tracking | 18 | 1.086 ms | 0.649 ms | 2.533 ms |
| Plate inference | 18 | 81.776 ms | 74.427 ms | 143.665 ms |
| OCR inference | 18 | 109.583 ms | 108.646 ms | 128.936 ms |

The measured effective rate was 2.606 sampled FPS. These figures are a local CPU
baseline, not a deployment target.

## Remaining production acceptance

Before deploying a camera, repeat the run with its natural video and the intended
Vietnamese plate checkpoint. Record model version/hash, day/night and vehicle-type
coverage, plate detection rate, OCR correctness, unknown rate, latency, and every
human correction. The temporary generic plate checkpoint above must not silently
become the production default.
