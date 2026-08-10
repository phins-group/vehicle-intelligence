"""Validate immutable edge artifacts before replacing the process with a worker."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from vehicle_intelligence.exceptions import VehicleIntelligenceError
from vehicle_intelligence.infrastructure.deployment import (
    apply_edge_environment,
    load_edge_manifest,
    resolve_edge_artifacts,
)


def main() -> None:
    manifest_value = os.getenv("VIP_EDGE_MANIFEST")
    model_root_value = os.getenv("VIP_EDGE_MODEL_ROOT", "/models")
    if not manifest_value:
        raise SystemExit("VIP_EDGE_MANIFEST is required")
    if len(sys.argv) < 2:
        raise SystemExit("edge entrypoint requires a worker command")
    try:
        manifest = load_edge_manifest(Path(manifest_value))
        artifacts = resolve_edge_artifacts(manifest, Path(model_root_value))
    except VehicleIntelligenceError as exc:
        raise SystemExit(str(exc)) from exc
    os.environ.update(apply_edge_environment(manifest, artifacts))
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
