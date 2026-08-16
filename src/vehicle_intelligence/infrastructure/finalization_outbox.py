"""Crash-safe filesystem outbox for complete vehicle-event finalization units."""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from vehicle_intelligence.application.finalization_outbox import (
    FinalizationMediaObject,
)
from vehicle_intelligence.application.ports import (
    MediaStorage,
    VehicleEventCodec,
    VehicleEventPublisher,
)
from vehicle_intelligence.config import FinalizationOutboxConfig
from vehicle_intelligence.domain import VehicleEvent
from vehicle_intelligence.exceptions import (
    EventBusError,
    EventContractError,
    FinalizationOutboxCorruptionError,
    FinalizationOutboxError,
    FinalizationOutboxFullError,
    FinalizationOutboxRetryableError,
    MediaStorageError,
    PersistenceError,
)

logger = logging.getLogger(__name__)

OUTBOX_SCHEMA_VERSION = 1
_ENTRY_PATTERN = re.compile(r"^event-[0-9]{8}T[0-9]{12}Z-[0-9a-f]{64}\.json$")
_TEMP_PATTERN = re.compile(r"^\.tmp-[0-9a-f]{32}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_FIELDS = (
    "snapshot_key",
    "vehicle_crop_key",
    "plate_crop_key",
)
_SUFFIX_BY_FIELD = {
    "snapshot_key": "snapshot.jpg",
    "vehicle_crop_key": "vehicle.jpg",
    "plate_crop_key": "plate.jpg",
}
_DELIVERY_ERRORS = (MediaStorageError, EventBusError, PersistenceError)
_RETRYABLE_DELIVERY_ERRORS = (*_DELIVERY_ERRORS, FinalizationOutboxRetryableError)


@dataclass(frozen=True, slots=True)
class _LoadedMedia:
    reference_field: str
    key: str
    data: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class _LoadedEntry:
    event: VehicleEvent
    media: tuple[_LoadedMedia, ...]
    file_sha256: str


class FilesystemFinalizationOutbox:
    """Persist event envelopes and JPEGs before attempting external delivery."""

    def __init__(
        self,
        config: FinalizationOutboxConfig,
        output_directory: str | Path,
        camera_id: str,
        codec: VehicleEventCodec,
        media_storage: MediaStorage,
        publisher: VehicleEventPublisher,
    ) -> None:
        if not camera_id.strip():
            raise ValueError("finalization outbox camera id cannot be empty")
        namespace = hashlib.sha256(camera_id.encode("utf-8")).hexdigest()
        output_root = Path(output_directory).expanduser().resolve()
        if output_root == Path(output_root.anchor):
            raise ValueError("finalization outbox output directory cannot be a filesystem root")
        self._root = output_root / "finalization-outbox" / namespace
        self._camera_id = camera_id
        self._config = config
        self._codec = codec
        self._media_storage = media_storage
        self._publisher = publisher
        self._delivery_lock = asyncio.Lock()
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._fatal_error: FinalizationOutboxError | None = None

    @property
    def directory(self) -> Path:
        """Return the camera-scoped durable directory for diagnostics."""

        return self._root

    async def initialize(self) -> None:
        if self._task is not None:
            return
        await asyncio.to_thread(self._initialize_sync)
        self._stop.clear()
        self._task = asyncio.create_task(
            self._run(),
            name=f"finalization-outbox-{self._camera_id}",
        )
        self._wake.set()
        await asyncio.sleep(0)

    async def stage(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error
        document = await asyncio.to_thread(self._encode_entry, event, media)
        if self._fatal_error is not None:
            raise self._fatal_error
        await asyncio.to_thread(
            self._stage_sync,
            event.id,
            event.occurred_at,
            document,
        )
        self._wake.set()

    async def replay_once(self) -> None:
        if self._fatal_error is not None:
            raise self._fatal_error
        async with self._delivery_lock:
            paths = await asyncio.to_thread(self._entry_paths_sync)
            for path in paths:
                try:
                    async with asyncio.timeout(self._config.delivery_timeout_seconds):
                        await self._deliver(path)
                except FinalizationOutboxCorruptionError:
                    raise
                except (*_RETRYABLE_DELIVERY_ERRORS, TimeoutError):
                    logger.warning(
                        "finalization outbox replay deferred",
                        extra={"camera_id": self._camera_id},
                        exc_info=True,
                    )
                    break

    async def close(self) -> None:
        task = self._task
        if task is None:
            return
        cleanup = asyncio.create_task(
            self._close_impl(task),
            name=f"finalization-outbox-close-{self._camera_id}",
        )
        caller_cancellation: asyncio.CancelledError | None = None
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as exc:
                caller_cancellation = exc
        cleanup_error: BaseException | None = None
        try:
            cleanup.result()
        except BaseException as exc:
            cleanup_error = exc
        if caller_cancellation is not None:
            if cleanup_error is not None:
                raise caller_cancellation from cleanup_error
            raise caller_cancellation
        if cleanup_error is not None:
            raise cleanup_error

    async def _close_impl(self, task: asyncio.Task[None]) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._config.delivery_timeout_seconds
        self._stop.set()
        self._wake.set()
        task.cancel()
        done, _pending = await asyncio.wait(
            {task},
            timeout=max(deadline - loop.time(), 0.0),
        )
        child_error: BaseException | None = None
        if task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except BaseException as exc:
                child_error = exc
            if self._task is task:
                self._task = None
        else:
            logger.error(
                "finalization outbox worker ignored cancellation before shutdown deadline",
                extra={"camera_id": self._camera_id},
            )
        remaining = max(deadline - loop.time(), 0.0)
        if remaining > 0:
            drain = asyncio.create_task(
                self.replay_once(),
                name=f"finalization-outbox-drain-{self._camera_id}",
            )
            drained, _pending = await asyncio.wait({drain}, timeout=remaining)
            if drain in drained:
                try:
                    drain.result()
                except FinalizationOutboxRetryableError:
                    logger.warning(
                        "finalization outbox shutdown drain deferred",
                        extra={"camera_id": self._camera_id},
                        exc_info=True,
                    )
                except BaseException as exc:
                    if child_error is None:
                        child_error = exc
                    else:
                        logger.error(
                            "finalization outbox shutdown drain failed",
                            extra={"camera_id": self._camera_id},
                            exc_info=(type(exc), exc, exc.__traceback__),
                        )
            else:
                drain.cancel()
                drain.add_done_callback(self._consume_task_result)
                logger.warning(
                    "finalization outbox shutdown drain reached its deadline",
                    extra={"camera_id": self._camera_id},
                )
        if child_error is not None:
            raise child_error

    @staticmethod
    def _consume_task_result(task: asyncio.Task[None]) -> None:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            task.result()

    async def _run(self) -> None:
        while not self._stop.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._wake.wait(),
                    timeout=self._config.replay_interval_seconds,
                )
            self._wake.clear()
            if self._stop.is_set():
                return
            try:
                await self.replay_once()
            except FinalizationOutboxCorruptionError as exc:
                self._fatal_error = exc
                logger.critical(
                    "finalization outbox integrity failure; replay stopped",
                    extra={"camera_id": self._camera_id},
                    exc_info=True,
                )
                return
            except FinalizationOutboxRetryableError:
                logger.warning(
                    "finalization outbox scan failed; replay will retry",
                    extra={"camera_id": self._camera_id},
                    exc_info=True,
                )
            except FinalizationOutboxError as exc:
                self._fatal_error = exc
                logger.critical(
                    "finalization outbox stopped after a non-retryable failure",
                    extra={"camera_id": self._camera_id},
                    exc_info=True,
                )
                return
            except Exception as exc:
                self._fatal_error = FinalizationOutboxError(
                    "finalization outbox worker stopped after an unexpected failure"
                )
                logger.exception(
                    "finalization outbox worker failed",
                    extra={"camera_id": self._camera_id},
                )
                self._fatal_error.__cause__ = exc
                return

    async def _deliver(self, path: Path) -> None:
        loaded = await asyncio.to_thread(self._load_entry_sync, path)
        for media in loaded.media:
            stored_key = await self._media_storage.put(
                media.key,
                media.data,
                media.content_type,
            )
            if stored_key != media.key:
                raise MediaStorageError("media storage returned a non-deterministic object key")
        await self._publisher.publish(loaded.event)
        await asyncio.to_thread(self._remove_entry_sync, path, loaded.file_sha256)

    def _initialize_sync(self) -> None:
        self._prepare_root_sync()
        try:
            with self._locked():
                entries = self._entry_paths_locked()
                self._fsync_directory(self._root)
                total_bytes = sum(path.stat().st_size for path in entries)
                if len(entries) > self._config.maximum_entries:
                    raise FinalizationOutboxFullError(
                        "durable finalization outbox exceeds configured entry capacity"
                    )
                if total_bytes > self._config.maximum_bytes:
                    raise FinalizationOutboxFullError(
                        "durable finalization outbox exceeds configured byte capacity"
                    )
                for path in entries:
                    self._load_entry_sync(path)
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot recover durable finalization outbox"
            ) from exc

    def _prepare_root_sync(self) -> None:
        try:
            output_root = self._root.parents[1]
            base = self._root.parent
            self._ensure_output_directory_tree(output_root)
            self._ensure_directory(base)
            self._fsync_directory(output_root)
            self._ensure_directory(self._root)
            self._fsync_directory(base)
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot initialize durable finalization outbox"
            ) from exc

    def _ensure_output_directory_tree(self, output_root: Path) -> None:
        missing: list[Path] = []
        cursor = output_root
        while not cursor.exists():
            if cursor.is_symlink():
                raise FinalizationOutboxCorruptionError(
                    "outbox output directory cannot traverse a symlink"
                )
            missing.append(cursor)
            cursor = cursor.parent
        if cursor.is_symlink() or not cursor.is_dir():
            raise FinalizationOutboxCorruptionError(
                "outbox output ancestor is not a safe directory"
            )
        for path in reversed(missing):
            try:
                path.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                created = False
            if path.is_symlink() or not path.is_dir():
                raise FinalizationOutboxCorruptionError(
                    "outbox output directory is not a safe directory"
                )
            if created:
                path.chmod(0o700)
                self._fsync_directory(path.parent)
                self._fsync_directory(path)

    @staticmethod
    def _ensure_directory(path: Path) -> None:
        if path.is_symlink():
            raise FinalizationOutboxCorruptionError("outbox directory cannot be a symlink")
        path.mkdir(mode=0o700, exist_ok=True)
        if not path.is_dir():
            raise FinalizationOutboxCorruptionError("outbox path is not a directory")
        path.chmod(0o700)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        lock_path = self._root / ".lock"
        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(lock_path, flags, 0o600)
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot lock durable finalization outbox"
            ) from exc
        try:
            try:
                metadata = os.fstat(descriptor)
                self._validate_owned_regular(metadata, "outbox lock")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                self._validate_owned_regular(os.fstat(descriptor), "outbox lock")
                os.fchmod(descriptor, 0o600)
            except OSError as exc:
                raise FinalizationOutboxRetryableError(
                    "cannot lock durable finalization outbox"
                ) from exc
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _stage_sync(
        self,
        event_id: str,
        occurred_at: datetime,
        document: bytes,
    ) -> Path:
        if len(document) > self._config.maximum_entry_bytes:
            raise FinalizationOutboxFullError(
                "finalization unit exceeds configured per-entry byte limit"
            )
        path = self._entry_path(event_id, occurred_at)
        with self._locked():
            entries = self._entry_paths_locked()
            digest_suffix = f"-{hashlib.sha256(event_id.encode('utf-8')).hexdigest()}.json"
            existing = next(
                (item for item in entries if item.name.endswith(digest_suffix)),
                None,
            )
            if existing is not None:
                if existing != path or self._read_entry_bytes(existing) != document:
                    raise FinalizationOutboxCorruptionError(
                        "existing finalization entry does not match deterministic event id"
                    )
                return existing
            total_bytes = sum(item.stat().st_size for item in entries)
            if len(entries) >= self._config.maximum_entries:
                raise FinalizationOutboxFullError(
                    "durable finalization outbox entry capacity reached"
                )
            if total_bytes + len(document) > self._config.maximum_bytes:
                raise FinalizationOutboxFullError(
                    "durable finalization outbox byte capacity reached"
                )
            temporary = self._root / f".tmp-{uuid.uuid4().hex}"
            try:
                self._write_file_sync(temporary, document)
                os.replace(temporary, path)
                self._fsync_directory(self._root)
            except OSError as exc:
                raise FinalizationOutboxRetryableError(
                    "cannot atomically stage durable finalization unit"
                ) from exc
            finally:
                with contextlib.suppress(OSError):
                    temporary.unlink()
            return path

    def _entry_paths_sync(self) -> tuple[Path, ...]:
        with self._locked():
            return self._entry_paths_locked()

    def _entry_paths_locked(self) -> tuple[Path, ...]:
        scanned = self._scan_root_sync()
        temporary = tuple(path for path in scanned if _TEMP_PATTERN.fullmatch(path.name))
        if temporary:
            try:
                for path in temporary:
                    metadata = path.lstat()
                    self._validate_owned_regular(metadata, "outbox temporary entry")
                    path.unlink()
                self._fsync_directory(self._root)
            except OSError as exc:
                raise FinalizationOutboxRetryableError(
                    "cannot remove abandoned outbox temporary entries"
                ) from exc
        entries = tuple(
            sorted(
                (path for path in scanned if _ENTRY_PATTERN.fullmatch(path.name)),
                key=lambda path: path.name,
            )
        )
        try:
            for path in entries:
                self._validate_owned_regular(path.lstat(), "outbox entry")
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot validate durable finalization outbox entries"
            ) from exc
        return entries

    def _scan_root_sync(self) -> tuple[Path, ...]:
        allowed = {".lock"}
        paths: list[Path] = []
        try:
            with os.scandir(self._root) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    if entry.name in allowed:
                        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                            raise FinalizationOutboxCorruptionError(
                                "outbox lock path is not a regular file"
                            )
                        continue
                    if not (
                        _ENTRY_PATTERN.fullmatch(entry.name) or _TEMP_PATTERN.fullmatch(entry.name)
                    ):
                        raise FinalizationOutboxCorruptionError(
                            "unrecognized file in durable finalization outbox"
                        )
                    if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                        raise FinalizationOutboxCorruptionError(
                            "outbox entry is not a regular file"
                        )
                    paths.append(path)
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot inspect durable finalization outbox"
            ) from exc
        return tuple(paths)

    def _load_entry_sync(self, path: Path) -> _LoadedEntry:
        raw = self._read_entry_bytes(path)
        file_sha256 = hashlib.sha256(raw).hexdigest()
        try:
            document = _strict_json_document(raw)
            _require_keys(document, {"schemaVersion", "payload", "payloadSha256"})
            schema_version = document["schemaVersion"]
            if type(schema_version) is not int or schema_version != OUTBOX_SCHEMA_VERSION:
                raise ValueError("unsupported finalization outbox schema version")
            payload = _require_dict(document["payload"], "payload")
            payload_checksum = _require_string(document["payloadSha256"], "payloadSha256")
            if not _SHA256_PATTERN.fullmatch(payload_checksum):
                raise ValueError("invalid finalization payload checksum")
            if hashlib.sha256(_canonical_json(payload)).hexdigest() != payload_checksum:
                raise ValueError("finalization payload checksum mismatch")
            _require_keys(payload, {"envelope", "media"})
            envelope = _require_string(payload["envelope"], "envelope")
            event = self._codec.decode(envelope)
            if event.camera.id != self._camera_id:
                raise ValueError("finalization event belongs to another camera")
            if path.name != self._entry_path(event.id, event.occurred_at).name:
                raise ValueError("finalization entry filename does not match event id")
            media = self._decode_media(payload["media"], event)
            return _LoadedEntry(event=event, media=media, file_sha256=file_sha256)
        except FinalizationOutboxCorruptionError:
            raise
        except (EventContractError, KeyError, TypeError, ValueError) as exc:
            raise FinalizationOutboxCorruptionError(
                "durable finalization entry failed integrity validation"
            ) from exc

    def _decode_media(self, raw_media: Any, event: VehicleEvent) -> tuple[_LoadedMedia, ...]:
        if not isinstance(raw_media, list):
            raise ValueError("finalization media must be a list")
        expected = {
            field: getattr(event.media, field)
            for field in _REFERENCE_FIELDS
            if getattr(event.media, field) is not None
        }
        if event.media.clip_key is not None:
            raise ValueError("finalization outbox does not support clip media")
        decoded: list[_LoadedMedia] = []
        seen: set[str] = set()
        for raw_item in raw_media:
            item = _require_dict(raw_item, "media item")
            _require_keys(
                item,
                {
                    "referenceField",
                    "key",
                    "contentType",
                    "size",
                    "sha256",
                    "dataBase64",
                },
            )
            field = _require_string(item["referenceField"], "referenceField")
            key = _require_string(item["key"], "key")
            content_type = _require_string(item["contentType"], "contentType")
            checksum = _require_string(item["sha256"], "sha256")
            encoded = _require_string(item["dataBase64"], "dataBase64")
            size = item["size"]
            if type(size) is not int or size <= 0:
                raise ValueError("finalization media size must be a positive integer")
            if field not in _REFERENCE_FIELDS or field in seen:
                raise ValueError("invalid or duplicate finalization media reference")
            if expected.get(field) != key:
                raise ValueError("finalization media key does not match event reference")
            if content_type != "image/jpeg":
                raise ValueError("finalization media content type must be image/jpeg")
            self._validate_media_key(event, field, key)
            try:
                data = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("invalid base64 finalization media") from exc
            if len(data) != size or hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError("finalization media checksum or size mismatch")
            decoded.append(_LoadedMedia(field, key, data, content_type))
            seen.add(field)
        if seen != set(expected):
            raise ValueError("finalization media set does not match event references")
        return tuple(decoded)

    @staticmethod
    def _validate_media_key(event: VehicleEvent, field: str, key: str) -> None:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or str(path) != key:
            raise ValueError("unsafe finalization media key")
        occurred = event.occurred_at.astimezone(UTC)
        prefix = f"vehicles/{occurred:%Y/%m/%d}/{event.camera.id}/{event.id}"
        if key != f"{prefix}/{_SUFFIX_BY_FIELD[field]}":
            raise ValueError("finalization media key is not deterministic")

    def _encode_entry(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> bytes:
        if event.camera.id != self._camera_id:
            raise FinalizationOutboxError("cannot stage an event for another camera")
        ordered = sorted(media, key=lambda item: item.reference_field)
        payload: dict[str, Any] = {
            "envelope": self._codec.encode(event),
            "media": [
                {
                    "referenceField": item.reference_field,
                    "key": item.key,
                    "contentType": item.content_type,
                    "size": len(item.data),
                    "sha256": hashlib.sha256(item.data).hexdigest(),
                    "dataBase64": base64.b64encode(item.data).decode("ascii"),
                }
                for item in ordered
            ],
        }
        document = {
            "schemaVersion": OUTBOX_SCHEMA_VERSION,
            "payload": payload,
            "payloadSha256": hashlib.sha256(_canonical_json(payload)).hexdigest(),
        }
        encoded = _canonical_json(document)
        # Validate the exact bytes before making them durable.
        temporary_name = self._entry_path(event.id, event.occurred_at)
        loaded = self._load_encoded_entry(encoded, temporary_name)
        if loaded.event.id != event.id:
            raise FinalizationOutboxCorruptionError("encoded finalization event id changed")
        return encoded

    def _load_encoded_entry(self, encoded: bytes, path: Path) -> _LoadedEntry:
        """Validate new bytes through the same parser without touching the filesystem."""

        try:
            document = _strict_json_document(encoded)
            payload = _require_dict(document["payload"], "payload")
            event = self._codec.decode(_require_string(payload["envelope"], "envelope"))
            media = self._decode_media(payload["media"], event)
            if path.name != self._entry_path(event.id, event.occurred_at).name:
                raise ValueError("finalization entry filename does not match event id")
            return _LoadedEntry(event, media, hashlib.sha256(encoded).hexdigest())
        except (EventContractError, KeyError, TypeError, ValueError) as exc:
            raise FinalizationOutboxCorruptionError(
                "new finalization unit failed integrity validation"
            ) from exc

    def _entry_path(self, event_id: str, occurred_at: datetime) -> Path:
        digest = hashlib.sha256(event_id.encode("utf-8")).hexdigest()
        occurred = occurred_at.astimezone(UTC)
        timestamp = (
            f"{occurred.year:04d}{occurred.month:02d}{occurred.day:02d}T"
            f"{occurred.hour:02d}{occurred.minute:02d}{occurred.second:02d}"
            f"{occurred.microsecond:06d}Z"
        )
        return self._root / f"event-{timestamp}-{digest}.json"

    def _read_entry_bytes(self, path: Path) -> bytes:
        flags = os.O_RDONLY
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(path, flags)
            metadata = os.fstat(descriptor)
            self._validate_owned_regular(metadata, "outbox entry")
            if stat.S_IMODE(metadata.st_mode) & 0o077:
                raise FinalizationOutboxCorruptionError("outbox entry permissions are too broad")
            if metadata.st_size > self._config.maximum_entry_bytes:
                raise FinalizationOutboxCorruptionError("outbox entry exceeds configured limit")
            with os.fdopen(descriptor, "rb", closefd=True) as stream:
                raw = stream.read(self._config.maximum_entry_bytes + 1)
            descriptor = None
        except FinalizationOutboxCorruptionError:
            raise
        except OSError as exc:
            raise FinalizationOutboxRetryableError(
                "cannot read durable finalization entry"
            ) from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
        if len(raw) > self._config.maximum_entry_bytes:
            raise FinalizationOutboxCorruptionError("outbox entry exceeds configured limit")
        return raw

    def _remove_entry_sync(self, path: Path, expected_sha256: str) -> None:
        with self._locked():
            try:
                metadata = path.lstat()
            except FileNotFoundError:
                return
            except OSError as exc:
                raise FinalizationOutboxRetryableError(
                    "cannot inspect delivered finalization entry"
                ) from exc
            self._validate_owned_regular(metadata, "delivered finalization entry")
            current = hashlib.sha256(self._read_entry_bytes(path)).hexdigest()
            if current != expected_sha256:
                raise FinalizationOutboxCorruptionError(
                    "finalization entry changed during delivery"
                )
            try:
                path.unlink()
                self._fsync_directory(self._root)
            except OSError as exc:
                raise FinalizationOutboxRetryableError(
                    "cannot acknowledge durable finalization entry"
                ) from exc

    @staticmethod
    def _write_file_sync(path: Path, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags, 0o600)
        try:
            FilesystemFinalizationOutbox._validate_owned_regular(
                os.fstat(descriptor),
                "new outbox entry",
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _validate_owned_regular(metadata: os.stat_result, name: str) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise FinalizationOutboxCorruptionError(f"{name} is not a regular file")
        if metadata.st_nlink != 1:
            raise FinalizationOutboxCorruptionError(f"{name} has an unsafe link count")
        if metadata.st_uid != os.geteuid():
            raise FinalizationOutboxCorruptionError(f"{name} has an unexpected owner")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _strict_json_document(raw: bytes) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    try:
        decoded = raw.decode("utf-8")
        value = json.loads(decoded, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FinalizationOutboxCorruptionError("outbox entry is not strict JSON") from exc
    return _require_dict(value, "document")


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _require_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _require_keys(value: dict[str, Any], expected: set[str]) -> None:
    if set(value) != expected:
        raise ValueError("outbox object contains missing or unknown fields")
