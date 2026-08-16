import asyncio
import os
import stat
from pathlib import Path

import pytest

from vehicle_intelligence.exceptions import MediaStorageError
from vehicle_intelligence.infrastructure.storage import local as local_storage_module
from vehicle_intelligence.infrastructure.storage.local import LocalMediaStorage


async def test_atomic_put_flushes_private_file_and_parent_directory(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "media"
    storage = LocalMediaStorage(root)
    flushed_types: list[str] = []
    real_fsync = local_storage_module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        mode = os.fstat(descriptor).st_mode
        flushed_types.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(descriptor)

    monkeypatch.setattr(local_storage_module.os, "fsync", recording_fsync)

    key = await storage.put("vehicles/gate/event/snapshot.jpg", b"jpeg", "image/jpeg")

    target = root / key
    assert target.read_bytes() == b"jpeg"
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(target.parent.stat().st_mode) == 0o700
    assert "file" in flushed_types
    assert flushed_types[-1] == "directory"
    assert not list(target.parent.glob(".*.tmp"))


async def test_failed_replace_preserves_previous_object_and_removes_temp(
    tmp_path,
    monkeypatch,
) -> None:
    root = tmp_path / "media"
    storage = LocalMediaStorage(root)
    key = "vehicles/gate/event/snapshot.jpg"
    await storage.put(key, b"old", "image/jpeg")

    def fail_replace(*args, **kwargs) -> None:
        del args, kwargs
        raise OSError("injected replace failure")

    monkeypatch.setattr(local_storage_module.os, "replace", fail_replace)

    with pytest.raises(MediaStorageError, match="cannot store"):
        await storage.put(key, b"new", "image/jpeg")

    target = root / key
    assert target.read_bytes() == b"old"
    assert not list(target.parent.glob(".*.tmp"))


async def test_concurrent_puts_use_distinct_temporary_files(tmp_path) -> None:
    root = tmp_path / "media"
    storage = LocalMediaStorage(root)
    key = "vehicles/gate/event/snapshot.jpg"

    await asyncio.gather(
        storage.put(key, b"first", "image/jpeg"),
        storage.put(key, b"second", "image/jpeg"),
    )

    target = root / key
    assert target.read_bytes() in {b"first", b"second"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert not list(target.parent.glob(".*.tmp"))


def test_bounded_read_holds_opened_inode_across_concurrent_replace(
    tmp_path,
    monkeypatch,
) -> None:
    target = tmp_path / "object.jpg"
    replacement = tmp_path / "replacement.jpg"
    target.write_bytes(b"small")
    replacement.write_bytes(b"x" * 4096)
    real_open = local_storage_module.os.open
    replaced = False

    def replacing_open(path, flags, *args, **kwargs):
        nonlocal replaced
        descriptor = real_open(path, flags, *args, **kwargs)
        if Path(path) == target and not replaced:
            replaced = True
            os.replace(replacement, target)
        return descriptor

    monkeypatch.setattr(local_storage_module.os, "open", replacing_open)

    assert LocalMediaStorage._read_bounded(target, 8) == b"small"
    with pytest.raises(MediaStorageError, match="read limit"):
        LocalMediaStorage._read_bounded(target, 8)


async def test_remove_flushes_parent_directory(tmp_path, monkeypatch) -> None:
    root = tmp_path / "media"
    storage = LocalMediaStorage(root)
    key = "vehicles/gate/event/snapshot.jpg"
    await storage.put(key, b"jpeg", "image/jpeg")
    flushed_directories = 0
    real_fsync = local_storage_module.os.fsync

    def recording_fsync(descriptor: int) -> None:
        nonlocal flushed_directories
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            flushed_directories += 1
        real_fsync(descriptor)

    monkeypatch.setattr(local_storage_module.os, "fsync", recording_fsync)

    await storage.remove(key)

    assert not (root / key).exists()
    assert flushed_directories == 1
