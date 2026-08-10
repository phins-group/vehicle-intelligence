"""MinIO object-storage adapter."""

from __future__ import annotations

import asyncio
from datetime import timedelta
from io import BytesIO
from pathlib import PurePosixPath

from vehicle_intelligence.config import MinioConfig
from vehicle_intelligence.domain import LifecycleReconcileResult
from vehicle_intelligence.exceptions import DependencyUnavailableError, MediaStorageError


class MinioMediaStorage:
    def __init__(self, config: MinioConfig) -> None:
        try:
            from minio import Minio
        except ImportError as exc:
            raise DependencyUnavailableError(
                "MinIO SDK is not installed; install the 'minio' extra"
            ) from exc
        self._client = Minio(
            config.endpoint,
            access_key=config.access_key.get_secret_value(),
            secret_key=config.secret_key.get_secret_value(),
            secure=config.secure,
            region=config.region,
        )
        self._public_client = Minio(
            config.public_endpoint or config.endpoint,
            access_key=config.access_key.get_secret_value(),
            secret_key=config.secret_key.get_secret_value(),
            secure=(config.public_secure if config.public_secure is not None else config.secure),
            region=config.region,
        )
        self._bucket = config.bucket
        self._bucket_ready = False
        self._bucket_lock = asyncio.Lock()

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        await self._ensure_bucket()
        try:
            await asyncio.to_thread(
                self._client.put_object,
                self._bucket,
                key,
                BytesIO(data),
                len(data),
                content_type=content_type,
            )
        except Exception as exc:
            raise MediaStorageError(f"cannot store MinIO object: {key}") from exc
        return key

    async def presign_get(self, key: str, expires: timedelta) -> str | None:
        try:
            await asyncio.to_thread(self._client.stat_object, self._bucket, key)
        except Exception as exc:
            if getattr(exc, "code", None) in {"NoSuchKey", "NoSuchObject"}:
                return None
            raise MediaStorageError(f"cannot inspect MinIO object: {key}") from exc
        try:
            return await asyncio.to_thread(
                self._public_client.presigned_get_object,
                self._bucket,
                key,
                expires=expires,
            )
        except Exception as exc:
            raise MediaStorageError(f"cannot authorize MinIO object: {key}") from exc

    async def exists(self, key: str) -> bool:
        try:
            await asyncio.to_thread(self._client.stat_object, self._bucket, key)
            return True
        except Exception as exc:
            if getattr(exc, "code", None) in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
                return False
            raise MediaStorageError(f"cannot inspect MinIO object: {key}") from exc

    async def get(self, key: str, maximum_bytes: int) -> bytes | None:
        if maximum_bytes <= 0:
            raise ValueError("maximum media read size must be positive")
        self._validate_key(key)
        try:
            stat = await asyncio.to_thread(self._client.stat_object, self._bucket, key)
            if stat.size > maximum_bytes:
                raise MediaStorageError("MinIO object exceeds configured read limit")
            return await asyncio.to_thread(self._read_bounded, key, maximum_bytes)
        except MediaStorageError:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) in {"NoSuchBucket", "NoSuchKey", "NoSuchObject"}:
                return None
            raise MediaStorageError(f"cannot read MinIO object: {key}") from exc

    async def remove(self, key: str) -> None:
        self._validate_key(key)
        try:
            await asyncio.to_thread(self._client.remove_object, self._bucket, key)
        except Exception as exc:
            raise MediaStorageError(f"cannot remove MinIO object: {key}") from exc

    async def reconcile_lifecycle(self, debug_expiry_days: int) -> LifecycleReconcileResult:
        await self._ensure_bucket()
        try:
            from minio.commonconfig import Filter
            from minio.lifecycleconfig import (
                Expiration,
                LifecycleConfig,
                Rule,
            )

            desired = [
                Rule(
                    "Enabled",
                    rule_filter=Filter(prefix="debug/"),
                    rule_id="vip-managed-debug-expiry",
                    expiration=Expiration(days=debug_expiry_days),
                ),
                Rule(
                    "Enabled",
                    rule_filter=Filter(prefix="temporary/"),
                    rule_id="vip-managed-temporary-expiry",
                    expiration=Expiration(days=debug_expiry_days),
                ),
            ]
            try:
                current = await asyncio.to_thread(
                    self._client.get_bucket_lifecycle,
                    self._bucket,
                )
            except Exception as exc:
                if getattr(exc, "code", None) != "NoSuchLifecycleConfiguration":
                    raise
                current = None
            existing = list(current.rules) if current is not None else []
            preserved = [
                rule for rule in existing if not (rule.rule_id or "").startswith("vip-managed-")
            ]
            combined = [*preserved, *desired]
            changed = existing != combined
            if changed:
                await asyncio.to_thread(
                    self._client.set_bucket_lifecycle,
                    self._bucket,
                    LifecycleConfig(combined),
                )
            return LifecycleReconcileResult(
                changed=changed,
                managed_rules=len(desired),
                preserved_rules=len(preserved),
            )
        except Exception as exc:
            raise MediaStorageError(f"cannot reconcile MinIO lifecycle: {self._bucket}") from exc

    async def _ensure_bucket(self) -> None:
        if self._bucket_ready:
            return
        async with self._bucket_lock:
            if self._bucket_ready:
                return
            try:
                exists = await asyncio.to_thread(self._client.bucket_exists, self._bucket)
                if not exists:
                    await asyncio.to_thread(self._client.make_bucket, self._bucket)
            except Exception as exc:
                raise MediaStorageError(f"cannot initialize MinIO bucket: {self._bucket}") from exc
            self._bucket_ready = True

    def _read_bounded(self, key: str, maximum_bytes: int) -> bytes:
        response = self._client.get_object(self._bucket, key)
        try:
            data = response.read(maximum_bytes + 1)
            if len(data) > maximum_bytes:
                raise MediaStorageError("MinIO object exceeds configured read limit")
            return data
        finally:
            response.close()
            response.release_conn()

    @staticmethod
    def _validate_key(key: str) -> None:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise MediaStorageError(f"unsafe MinIO media key: {key}")
