"""Atomic local development media storage."""

from __future__ import annotations

import asyncio
import contextlib
import os
import stat
import uuid
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
            await asyncio.to_thread(self._remove_sync, target)
        except OSError as exc:
            raise MediaStorageError(f"cannot remove local media object: {key}") from exc

    def _resolve_key(self, key: str) -> Path:
        path = PurePosixPath(key)
        if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != key:
            raise MediaStorageError(f"unsafe media key: {key}")
        target = self._root.joinpath(*path.parts)
        if not target.is_relative_to(self._root):
            raise MediaStorageError(f"media key escapes storage root: {key}")
        cursor = self._root
        for part in path.parts:
            cursor /= part
            if cursor.is_symlink():
                raise MediaStorageError(f"media key traverses a symlink: {key}")
        return target

    def _write_atomic(self, target: Path, data: bytes) -> None:
        self._ensure_directory_tree(target.parent)
        parent_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        parent_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        parent_descriptor = os.open(target.parent, parent_flags)
        temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"
        temporary_descriptor: int | None = None
        try:
            parent_metadata = os.fstat(parent_descriptor)
            if not stat.S_ISDIR(parent_metadata.st_mode):
                raise OSError("local media parent is not a directory")
            file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            file_flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            temporary_descriptor = os.open(
                temporary_name,
                file_flags,
                0o600,
                dir_fd=parent_descriptor,
            )
            metadata = os.fstat(temporary_descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_nlink != 1
                or metadata.st_uid != os.geteuid()
            ):
                raise OSError("local media temporary file failed ownership validation")
            os.fchmod(temporary_descriptor, 0o600)
            with os.fdopen(temporary_descriptor, "wb", closefd=False) as stream:
                stream.write(data)
                stream.flush()
                os.fsync(temporary_descriptor)
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        finally:
            if temporary_descriptor is not None:
                os.close(temporary_descriptor)
            with contextlib.suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            os.close(parent_descriptor)

    def _ensure_directory_tree(self, directory: Path) -> None:
        if not directory.is_relative_to(self._root):
            raise MediaStorageError("local media parent escapes storage root")
        missing: list[Path] = []
        cursor = directory
        while not cursor.exists():
            if cursor.is_symlink():
                raise MediaStorageError("local media parent cannot be a symlink")
            missing.append(cursor)
            cursor = cursor.parent
        if cursor.is_symlink() or not cursor.is_dir():
            raise MediaStorageError("local media ancestor is not a safe directory")
        for path in reversed(missing):
            try:
                path.mkdir(mode=0o700)
                created = True
            except FileExistsError:
                created = False
            if path.is_symlink() or not path.is_dir():
                raise MediaStorageError("local media parent is not a safe directory")
            if created:
                path.chmod(0o700)
                self._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_bounded(target: Path, maximum_bytes: int) -> bytes | None:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags)
        except FileNotFoundError:
            return None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise MediaStorageError("local media object is not a regular file")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                data = stream.read(maximum_bytes + 1)
        finally:
            os.close(descriptor)
        if len(data) > maximum_bytes:
            raise MediaStorageError("local media object exceeds configured read limit")
        return data

    @staticmethod
    def _remove_sync(target: Path) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            parent_descriptor = os.open(target.parent, flags)
        except FileNotFoundError:
            return
        try:
            try:
                os.unlink(target.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                return
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
