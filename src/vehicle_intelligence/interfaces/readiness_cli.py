"""Machine-readable static production readiness command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vehicle_intelligence.application.production_readiness import (
    ReadinessStatus,
    assess_production_readiness,
)
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import VehicleIntelligenceError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate secret-safe static production deployment prerequisites"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--base-directory",
        type=Path,
        default=Path.cwd(),
        help="Base directory for relative model artifact paths",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Return a non-zero status when warnings remain",
    )
    parser.add_argument("--output", type=Path, help="Atomically write the JSON report")
    return parser


def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    report = assess_production_readiness(
        settings,
        base_directory=args.base_directory,
    )
    rendered = json.dumps(report.to_document(), indent=2, sort_keys=True)
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        temporary.replace(output)
    print(rendered)
    warnings = report.counts[ReadinessStatus.WARN.value]
    return 0 if report.ready and (not args.strict_warnings or warnings == 0) else 4


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
