# Vision Pipeline

## Per-frame path

The default vehicle-first path is:

1. OpenCV decodes the source and derives a media timestamp from frame position.
2. The sampler keeps at most the configured inference FPS; skipped frames are
   never queued.
3. The vehicle detector returns only configured vehicle classes and confidence.
4. ByteTrack associates detections and returns camera-local track IDs.
5. The pipeline updates track trajectory/type evidence and scores the best full
   vehicle frame.
6. Plate detection runs on the clipped vehicle crop, not the whole image.
7. Every plate crop receives blur, brightness, contrast, resolution, aspect-angle
   proxy, and detector-confidence scores.
8. Crops below `plate_quality.minimum` are not sent to OCR.
9. OCR runs on the original useful crop and, only when indicated, an adaptive
   processed variant. The best valid/confident result wins.
10. Raw OCR is normalized separately and appended as a `PlateObservation`.
11. When live monitoring is enabled, matching overlay metadata and a copied
    source frame enter a one-slot, latest-only reporter. Resize/JPEG encoding and
    best-effort publication run in the background and never change inference or
    event semantics.

With `vision.plate_only: true` or `--plate-only`, steps 3–6 are replaced by one
full-frame plate inference followed by ByteTrack association of canonical plate
boxes. Each plate track independently receives quality filtering, OCR,
temporal voting, timeout/EOF finalization, a full-frame snapshot, and a best
plate crop. No vehicle detector is loaded or called; the canonical event uses
unknown vehicle evidence and has no vehicle crop or vehicle-detector trace.

## Detector runtimes

The pipeline receives `VehicleDetector` and `PlateDetector` ports. A
configuration-driven infrastructure factory can supply YOLO/Ultralytics,
PicoDet, or the existing YOLO ONNX Runtime adapter, including
CoreML/CUDA/TensorRT execution providers. Requested acceleration must exist;
there is no silent CPU benchmark fallback. Provider output is normalized to the
same domain detections before tracking or plate quality logic. Vehicle providers
return `Detection`; plate providers return the existing framework-neutral
`PlateDetection`, which additionally permits optional perspective corners. See
[Model Runtime Optimization](MODEL_OPTIMIZATION.md).

Vehicle and plate providers are selected independently. Switching either side
does not change pipeline source:

```yaml
vision:
  vehicle_detection:
    provider: picodet
    model_path: models/vehicle-picodet.onnx
    model_classes: [car, motorcycle, bus, truck]
    classes: [car, motorcycle, bus, truck]
  plate_detection:
    provider: yolo
    model_path: models/vietnam-plate.pt
```

The inverse combination is also supported. `yolo` is an alias for the retained
`ultralytics` behavior, so existing configurations remain compatible.

## Adaptive preprocessing

The preprocessor can resize, denoise, apply CLAHE, and sharpen. It always keeps
the original as a candidate. Enhancement is added only when measured contrast or
sharpness falls below configured values. Perspective correction is applied when
the plate detector provides four corners; an axis-aligned detector correctly
falls back to the crop without pretending that geometry is known.

## Quality score

Each component is normalized to `[0, 1]`. `total_score` is a configured weighted
mean:

```text
sharpness + brightness + contrast + resolution + angle + detector confidence
```

Weights are validated to sum to a positive value and normalized during
calculation. Minimum plate dimensions are a hard eligibility gate because an
apparently sharp but tiny crop is not useful OCR evidence.

## Temporal aggregation

For a finalized track:

1. Group candidates whose compact forms are within the configured edit distance.
2. Score clusters using candidate frequency and observation evidence.
3. Align same-length members and compute position-wise weighted character votes.
4. Normalize the consensus again to enforce plate structure.
5. Return a bounded aggregate confidence and supporting observation count.

The aggregate may exceed a noisy individual observation because repeated,
independent agreement increases support; it remains capped below 1.0.

## Finalization

Finalization triggers are timeout, leaving ROI, optional line trigger, and EOF.
The finalizer selects the best vehicle/plate artifacts, computes direction from
trajectory, aggregates plate evidence, emits one event, writes media, and then
persists the event. Unknown plates still create vehicle events.

## Error isolation

Detector/tracker errors are fatal for the current video run because subsequent
association would be unreliable. Plate detector and OCR errors are logged with
camera/track context and skip that observation; tracking continues. This policy
is explicit so silent partial inference cannot masquerade as success.

The optional live-preview path is independently isolated. A full pending slot
drops the older preview, and encoder, contract, timeout, or broker failures are
counted without failing the vision run. See [Live Monitor](LIVE_MONITOR.md) for
the bounded transport and UI contract.
