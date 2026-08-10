#!/usr/bin/env python3
"""Export a real Ultralytics detector and write an integrity manifest."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a versioned detector artifact")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--format", choices=("onnx", "engine"), required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--device")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--simplify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--manifest", type=Path)
    return parser


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = build_parser().parse_args()
    if args.image_size < 1 or args.opset < 12 or args.workspace_gb <= 0:
        raise SystemExit("invalid export bounds")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    checkpoint_hash = sha256_file(checkpoint)
    expected = (args.expected_checkpoint_sha256 or "").lower().removeprefix("sha256:")
    if expected and checkpoint_hash != expected:
        raise SystemExit("checkpoint SHA-256 mismatch")
    if args.format == "engine" and (args.device or "cpu").lower() == "cpu":
        raise SystemExit("TensorRT engine export requires an explicit CUDA device")
    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("install the 'vision' extra before model export") from exc
    model = YOLO(str(checkpoint))
    options: dict[str, object] = {
        "format": args.format,
        "imgsz": args.image_size,
        "dynamic": args.dynamic,
        "simplify": args.simplify,
        "half": args.half,
    }
    if args.device is not None:
        options["device"] = args.device
    if args.format == "onnx":
        options["opset"] = args.opset
    else:
        options["workspace"] = args.workspace_gb
    exported = Path(str(model.export(**options))).resolve()
    if not exported.is_file():
        raise SystemExit(f"export did not produce an artifact: {exported}")
    validation: dict[str, object] = {"exportedByUltralytics": True}
    if args.format == "onnx":
        try:
            import onnx
            import onnxruntime as ort
        except ImportError as exc:
            raise SystemExit("install the 'optimization' extra to validate ONNX export") from exc
        onnx.checker.check_model(onnx.load(str(exported)))
        session = ort.InferenceSession(str(exported), providers=["CPUExecutionProvider"])
        validation.update(
            {
                "onnxChecker": "passed",
                "runtimeLoad": "passed",
                "inputs": [item.name for item in session.get_inputs()],
                "outputs": [item.name for item in session.get_outputs()],
            }
        )
    manifest: dict[str, object] = {
        "schemaVersion": 1,
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "model": {"name": args.model_name, "version": args.model_version},
        "source": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
        },
        "artifact": {
            "format": args.format,
            "path": str(exported),
            "sha256": sha256_file(exported),
            "sizeBytes": exported.stat().st_size,
        },
        "export": {
            "ultralyticsVersion": ultralytics.__version__,
            "imageSize": args.image_size,
            "dynamic": args.dynamic,
            "half": args.half,
            "device": args.device,
            "opset": args.opset if args.format == "onnx" else None,
        },
        "validation": validation,
    }
    manifest_path = args.manifest or exported.with_suffix(exported.suffix + ".manifest.json")
    _atomic_json(manifest_path.resolve(), manifest)
    print(json.dumps({"artifact": str(exported), "manifest": str(manifest_path.resolve())}))


if __name__ == "__main__":
    main()
