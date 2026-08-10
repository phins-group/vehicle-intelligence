"""Controlled camera RTSP credential key-rotation command."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict

from vehicle_intelligence.application.audit import AuditRecord, AuditService
from vehicle_intelligence.application.credential_rotation import (
    CameraCredentialRotationService,
)
from vehicle_intelligence.config import load_settings
from vehicle_intelligence.domain import (
    AuditAction,
    AuditResourceType,
    AuthenticationMethod,
    Principal,
    UserRole,
)
from vehicle_intelligence.exceptions import ConfigurationError
from vehicle_intelligence.infrastructure.persistence.audit_mongo import MongoAuditLogRepository
from vehicle_intelligence.infrastructure.persistence.camera_mongo import MongoCameraRepository
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher
from vehicle_intelligence.logging_config import configure_logging


async def _run(args: argparse.Namespace) -> int:
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    if not settings.mongodb.enabled:
        raise ConfigurationError("credential rotation requires MongoDB")
    cipher = AesGcmCredentialCipher.from_config(settings.security)
    runtime = MongoRuntime(settings.mongodb)
    await runtime.initialize()
    cameras = MongoCameraRepository(runtime, cipher)
    audits = AuditService(MongoAuditLogRepository(runtime))
    try:
        await cameras.ensure_indexes()
        await audits.initialize()
        service = CameraCredentialRotationService(cameras, cipher)
        report = await service.rotate(
            batch_size=args.batch_size,
            maximum_cameras=args.limit,
            dry_run=args.dry_run,
        )
        await audits.record(
            AuditRecord(
                principal=Principal(
                    id="credential-rotation-worker",
                    display_name="Credential Rotation Worker",
                    role=UserRole.ADMIN,
                    authentication_method=AuthenticationMethod.SYSTEM,
                ),
                action=AuditAction.CAMERA_CREDENTIALS_ROTATED,
                resource_type=AuditResourceType.CAMERA,
                resource_id="*",
                request_id=f"credential-rotation:{report.active_key_id}",
                metadata=asdict(report),
            )
        )
        print(json.dumps(asdict(report), sort_keys=True))
        return 0 if report.conflicts == 0 else 2
    finally:
        await audits.close()
        await cameras.close()
        await runtime.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rotate encrypted camera RTSP credentials")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    raise SystemExit(asyncio.run(_run(_parser().parse_args())))


if __name__ == "__main__":
    main()
