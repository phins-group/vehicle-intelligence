#!/usr/bin/env python3
"""Create a deployable edge manifest from two real model artifacts."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from vehicle_intelligence.infrastructure.deployment import (
    EdgeArtifact,
    EdgeDeploymentManifest,
    resolve_edge_artifacts,
)
from vehicle_intelligence.infrastructure.vision.model_artifact import sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an integrity-bound edge manifest")
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--config-version", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    for role in ("vehicle", "plate"):
        parser.add_argument(f"--{role}-model", type=Path, required=True)
        parser.add_argument(
            f"--{role}-provider",
            choices=("onnxruntime", "tensorrt"),
            required=True,
        )
        parser.add_argument(f"--{role}-model-name", required=True)
        parser.add_argument(f"--{role}-model-version", required=True)
        parser.add_argument(f"--{role}-execution-provider", action="append", default=[])
    return parser


def _artifact(args: argparse.Namespace, role: str, root: Path) -> EdgeArtifact:
    path = getattr(args, f"{role}_model").expanduser().resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"{role} model must be inside --model-root") from exc
    if not path.is_file():
        raise SystemExit(f"{role} model does not exist: {path}")
    return EdgeArtifact(
        role=role,
        relativePath=relative.as_posix(),
        provider=getattr(args, f"{role}_provider"),
        modelName=getattr(args, f"{role}_model_name"),
        modelVersion=getattr(args, f"{role}_model_version"),
        sha256=sha256_file(path),
        sizeBytes=path.stat().st_size,
        executionProviders=getattr(args, f"{role}_execution_provider"),
    )


def main() -> None:
    args = build_parser().parse_args()
    root = args.model_root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"model root does not exist: {root}")
    manifest = EdgeDeploymentManifest(
        schemaVersion=1,
        nodeId=args.node_id,
        configVersion=args.config_version,
        createdAt=datetime.now(UTC),
        artifacts=[_artifact(args, role, root) for role in ("vehicle", "plate")],
    )
    resolve_edge_artifacts(manifest, root, check_runtime=False)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest.model_dump(by_alias=True, mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output)


if __name__ == "__main__":
    main()
