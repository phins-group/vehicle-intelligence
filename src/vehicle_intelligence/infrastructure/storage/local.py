"""Atomic local development media storage."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path, PurePosixPath

from vehicle_intelligence.exceptions import MediaStorageError


class LocalMediaStorage:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).expanduser().resolve()

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        del content_type
        target = self._resolve_key(key)
        try:
            await asyncio.to_thread(self._write_atomic, target, data)
        except OSError as exc:
            raise MediaStorageError(f"cannot store local media object: {key}") from exc
        return key

    async def exists(self, key: str) -> bool:
        target = self._resolve_key(key)
        return await asyncio.to_thread(target.is_file)

    async def get(self, key: str, maximum_bytes: int) -> bytes | None:
        if maximum_bytes <= 0:
            raise ValueError("maximum media read size must be positive")
        target = self._resolve_key(key)
        try:
            return await asyncio.to_thread(self._read_bounded, target, maximum_bytes)
        except OSError as exc:
            raise MediaStorageError(f"cannot read local media object: {key}") from exc

    async def remove(self, key: str) -> None:
        target = self._resolve_key(key)
        try:
            await asyncio.to_thread(target.unlink, missing_ok=True)
        except OSError as exc:
            raise MediaStorageError(f"cannot remove local media object: {key}") from exc

    def _resolve_key(self, key: str) -> Path:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise MediaStorageError(f"unsafe media key: {key}")
        target = self._root.joinpath(*path.parts).resolve()
        if not target.is_relative_to(self._root):
            raise MediaStorageError(f"media key escapes storage root: {key}")
        return target

    @staticmethod
    def _write_atomic(target: Path, data: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_bounded(target: Path, maximum_bytes: int) -> bytes | None:
        if not target.is_file():
            return None
        if target.stat().st_size > maximum_bytes:
            raise MediaStorageError("local media object exceeds configured read limit")
        data = target.read_bytes()
        if len(data) > maximum_bytes:
            raise MediaStorageError("local media object exceeds configured read limit")
        return data
