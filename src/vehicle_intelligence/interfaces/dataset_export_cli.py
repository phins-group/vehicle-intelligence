"""Bounded OCR feedback dataset export command."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from vehicle_intelligence.application.dataset_export import OCRDatasetExportService
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError, VehicleIntelligenceError
from vehicle_intelligence.infrastructure.persistence.mongo import MongoVehicleEventRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.persistence.review_mongo import (
    MongoDatasetSampleRepository,
)
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage
from vehicle_intelligence.infrastructure.vision.opencv import OpenCVDatasetImageTranscoder
from vehicle_intelligence.logging_config import configure_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export reviewed OCR samples for retraining")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--export-id")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser


async def run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if not settings.mongodb.enabled:
        raise ConfigurationError("dataset export requires mongodb.enabled=true")
    export_config = settings.dataset_export
    if args.output is not None:
        export_config = export_config.model_copy(update={"output_directory": args.output})
    export_id = args.export_id or (
        f"ocr-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    )
    runtime = MongoRuntime(settings.mongodb)
    await runtime.initialize()
    events = MongoVehicleEventRepository(runtime)
    samples = MongoDatasetSampleRepository(runtime)
    media = (
        MinioMediaStorage(settings.minio)
        if settings.storage.backend == "minio"
        else LocalMediaStorage(settings.storage.output_directory)
    )
    service = OCRDatasetExportService(
        export_config,
        samples,
        events,
        media,
        OpenCVDatasetImageTranscoder(),
    )
    try:
        await events.ensure_indexes()
        await samples.ensure_indexes()
        result = await service.export(export_id, args.limit)
        print(
            json.dumps(
                {
                    "exportId": result.export_id,
                    "directory": str(result.directory) if result.directory else None,
                    "manifestSha256": result.manifest_sha256,
                    "exportedCount": result.exported_count,
                    "failedCount": result.failed_count,
                    "splitCounts": result.split_counts,
                    "reused": result.reused,
                },
                sort_keys=True,
            )
        )
        return 0 if result.failed_count == 0 else 3
    finally:
        try:
            await service.close()
        finally:
            try:
                if isinstance(media, MinioMediaStorage):
                    await media.close()
            finally:
                await runtime.close()


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(asyncio.run(run(parser.parse_args())))
    except VehicleIntelligenceError as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
