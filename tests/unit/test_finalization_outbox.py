import asyncio
import json
import os
import stat
import uuid
from dataclasses import replace
from datetime import timedelta

import pytest

from vehicle_intelligence.application.finalization_outbox import (
    FinalizationMediaObject,
)
from vehicle_intelligence.config import FinalizationOutboxConfig
from vehicle_intelligence.domain import MediaReferences, VehicleEvent
from vehicle_intelligence.exceptions import (
    EventBusError,
    FinalizationOutboxCorruptionError,
    FinalizationOutboxError,
    FinalizationOutboxFullError,
    FinalizationOutboxRetryableError,
    MediaStorageError,
    PersistenceError,
)
from vehicle_intelligence.infrastructure.finalization_outbox import (
    FilesystemFinalizationOutbox,
)
from vehicle_intelligence.infrastructure.messaging.codec import JsonEventEnvelopeCodec


class _MediaStorage:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.objects: dict[str, bytes] = {}
        self.attempted = asyncio.Event()

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        assert content_type == "image/jpeg"
        self.attempted.set()
        if self.fail:
            raise MediaStorageError("injected media outage")
        self.objects[key] = data
        return key


class _Publisher:
    def __init__(
        self,
        persisted_ids: set[str] | None = None,
        *,
        fail_after_save: bool = False,
    ) -> None:
        self.persisted_ids = persisted_ids if persisted_ids is not None else set()
        self.fail_after_save = fail_after_save
        self.calls = 0
        self.published_ids: list[str] = []
        self.attempted = asyncio.Event()

    async def initialize(self) -> None:
        return None

    async def publish(self, event: VehicleEvent) -> bool:
        self.calls += 1
        self.published_ids.append(event.id)
        self.attempted.set()
        created = event.id not in self.persisted_ids
        self.persisted_ids.add(event.id)
        if self.fail_after_save:
            raise EventBusError("injected crash after publish")
        return created

    async def close(self) -> None:
        return None


class _PersistenceFailPublisher(_Publisher):
    async def publish(self, event: VehicleEvent) -> bool:
        self.calls += 1
        self.published_ids.append(event.id)
        self.attempted.set()
        raise PersistenceError("injected persistence outage")


class _HungMediaStorage:
    def __init__(self) -> None:
        self.started = asyncio.Event()

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        del key, data, content_type
        self.started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


class _CancelThenSucceedMediaStorage:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        assert content_type == "image/jpeg"
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        self.objects[key] = data
        return key


class _SelectiveFailMediaStorage:
    def __init__(self, failing_key: str) -> None:
        self._failing_key = failing_key
        self.attempted_keys: list[str] = []

    async def put(self, key: str, data: bytes, content_type: str) -> str:
        del data
        assert content_type == "image/jpeg"
        self.attempted_keys.append(key)
        if key == self._failing_key:
            raise MediaStorageError("injected head-of-line outage")
        return key


def _config(**updates) -> FinalizationOutboxConfig:
    values = {
        "maximum_entries": 10,
        "maximum_bytes": 1024 * 1024,
        "maximum_entry_bytes": 64 * 1024,
        "delivery_timeout_seconds": 0.2,
        "replay_interval_seconds": 60.0,
    }
    values.update(updates)
    return FinalizationOutboxConfig(**values)


def _event_and_media(
    sample_event: VehicleEvent,
    event_id: str,
) -> tuple[VehicleEvent, tuple[FinalizationMediaObject, ...]]:
    key = (
        f"vehicles/{sample_event.occurred_at:%Y/%m/%d}/"
        f"{sample_event.camera.id}/{event_id}/snapshot.jpg"
    )
    event = replace(
        sample_event,
        id=event_id,
        track_id=f"{sample_event.camera.id}:outbox:{event_id}",
        media=MediaReferences(snapshot_key=key),
    )
    media = (
        FinalizationMediaObject(
            reference_field="snapshot_key",
            key=key,
            data=f"jpeg:{event_id}".encode(),
        ),
    )
    return event, media


def _outbox(
    tmp_path,
    sample_event: VehicleEvent,
    media_storage,
    publisher: _Publisher,
    *,
    config: FinalizationOutboxConfig | None = None,
) -> FilesystemFinalizationOutbox:
    return FilesystemFinalizationOutbox(
        config or _config(),
        tmp_path,
        sample_event.camera.id,
        JsonEventEnvelopeCodec(),
        media_storage,
        publisher,
    )


async def test_media_outage_is_replayed_after_restart(tmp_path, sample_event) -> None:
    event, media = _event_and_media(sample_event, "evt_media_restart")
    unavailable = _MediaStorage(fail=True)
    first = _outbox(tmp_path, sample_event, unavailable, _Publisher())
    await first.initialize()
    await first.stage(event, media)
    await asyncio.wait_for(unavailable.attempted.wait(), timeout=1)
    entries = list(first.directory.glob("event-*.json"))
    assert len(entries) == 1
    assert first.directory.stat().st_mode & 0o777 == 0o700
    assert entries[0].stat().st_mode & 0o777 == 0o600
    await first.close()

    recovered_media = _MediaStorage()
    recovered_publisher = _Publisher()
    restarted = _outbox(tmp_path, sample_event, recovered_media, recovered_publisher)
    await restarted.initialize()
    await restarted.replay_once()

    assert recovered_media.objects[media[0].key] == media[0].data
    assert recovered_publisher.persisted_ids == {event.id}
    assert not list(restarted.directory.glob("event-*.json"))
    await restarted.close()


async def test_crash_after_publish_replays_with_event_id_idempotency(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_publish_restart")
    persisted_ids: set[str] = set()
    crashing = _Publisher(persisted_ids, fail_after_save=True)
    first = _outbox(tmp_path, sample_event, _MediaStorage(), crashing)
    await first.initialize()
    await first.stage(event, media)
    await asyncio.wait_for(crashing.attempted.wait(), timeout=1)
    assert persisted_ids == {event.id}
    assert len(list(first.directory.glob("event-*.json"))) == 1
    await first.close()

    replay = _Publisher(persisted_ids)
    restarted = _outbox(tmp_path, sample_event, _MediaStorage(), replay)
    await restarted.initialize()
    await restarted.replay_once()

    assert persisted_ids == {event.id}
    assert replay.calls >= 1
    assert not list(restarted.directory.glob("event-*.json"))
    await restarted.close()


async def test_persistence_error_keeps_durable_entry(tmp_path, sample_event) -> None:
    event, media = _event_and_media(sample_event, "evt_persistence_error")
    publisher = _PersistenceFailPublisher()
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(), publisher)
    await outbox.initialize()
    await outbox.stage(event, media)
    await asyncio.wait_for(publisher.attempted.wait(), timeout=1)

    assert len(list(outbox.directory.glob("event-*.json"))) == 1
    await outbox.close()
    assert len(list(outbox.directory.glob("event-*.json"))) == 1


async def test_capacity_rejects_new_entry_without_dropping_oldest(
    tmp_path,
    sample_event,
) -> None:
    first_event, first_media = _event_and_media(sample_event, "evt_capacity_first")
    second_event, second_media = _event_and_media(sample_event, "evt_capacity_second")
    unavailable = _MediaStorage(fail=True)
    outbox = _outbox(
        tmp_path,
        sample_event,
        unavailable,
        _Publisher(),
        config=_config(maximum_entries=1),
    )
    await outbox.initialize()
    await outbox.stage(first_event, first_media)

    with pytest.raises(FinalizationOutboxFullError, match="entry capacity"):
        await outbox.stage(second_event, second_media)

    entries = list(outbox.directory.glob("event-*.json"))
    assert len(entries) == 1
    staged = entries[0].read_text(encoding="utf-8")
    assert first_event.id in staged
    assert second_event.id not in staged
    await outbox.close()


async def test_byte_capacity_rejects_new_entry_without_eviction(
    tmp_path,
    sample_event,
) -> None:
    first_event, first_media = _event_and_media(sample_event, "evt_bytes_first")
    second_event, second_media = _event_and_media(sample_event, "evt_bytes_second")
    large_jpeg = b"x" * 600_000
    first_media = (replace(first_media[0], data=large_jpeg),)
    second_media = (replace(second_media[0], data=large_jpeg),)
    outbox = _outbox(
        tmp_path,
        sample_event,
        _MediaStorage(fail=True),
        _Publisher(),
        config=_config(
            maximum_bytes=1024 * 1024,
            maximum_entry_bytes=1024 * 1024,
        ),
    )
    await outbox.initialize()
    await outbox.stage(first_event, first_media)

    with pytest.raises(FinalizationOutboxFullError, match="byte capacity"):
        await outbox.stage(second_event, second_media)

    [entry] = outbox.directory.glob("event-*.json")
    staged = entry.read_text(encoding="utf-8")
    assert first_event.id in staged
    assert second_event.id not in staged
    await outbox.close()


async def test_tampered_entry_fails_closed_on_restart(tmp_path, sample_event) -> None:
    event, media = _event_and_media(sample_event, "evt_tampered")
    unavailable = _MediaStorage(fail=True)
    first = _outbox(tmp_path, sample_event, unavailable, _Publisher())
    await first.initialize()
    await first.stage(event, media)
    await asyncio.wait_for(unavailable.attempted.wait(), timeout=1)
    await first.close()

    [entry] = first.directory.glob("event-*.json")
    document = json.loads(entry.read_text(encoding="utf-8"))
    document["payload"]["media"][0]["key"] = "vehicles/tampered.jpg"
    entry.write_text(json.dumps(document), encoding="utf-8")

    restarted = _outbox(tmp_path, sample_event, _MediaStorage(), _Publisher())
    with pytest.raises(FinalizationOutboxCorruptionError, match="integrity"):
        await restarted.initialize()
    assert entry.is_file()


async def test_hung_delivery_does_not_block_stage_and_close_is_cancellable(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_hung_delivery")
    hung = _HungMediaStorage()
    outbox = _outbox(tmp_path, sample_event, hung, _Publisher())
    await outbox.initialize()

    await asyncio.wait_for(outbox.stage(event, media), timeout=0.5)
    await asyncio.wait_for(hung.started.wait(), timeout=0.5)
    await asyncio.wait_for(outbox.close(), timeout=0.5)

    assert len(list(outbox.directory.glob("event-*.json"))) == 1


async def test_close_performs_bounded_final_delivery_for_finite_runs(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_close_drain")
    storage = _CancelThenSucceedMediaStorage()
    publisher = _Publisher()
    outbox = _outbox(tmp_path, sample_event, storage, publisher)
    await outbox.initialize()
    await outbox.stage(event, media)
    await asyncio.wait_for(storage.started.wait(), timeout=0.5)

    await asyncio.wait_for(outbox.close(), timeout=0.5)

    assert storage.objects[media[0].key] == media[0].data
    assert publisher.persisted_ids == {event.id}
    assert not list(outbox.directory.glob("event-*.json"))


async def test_replay_scan_io_failure_is_retried_without_killing_worker(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_scan_retry")
    publisher = _Publisher()
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(), publisher)
    original_scan = outbox._entry_paths_sync
    scan_attempts = 0

    def flaky_scan():
        nonlocal scan_attempts
        scan_attempts += 1
        if scan_attempts == 1:
            raise FinalizationOutboxRetryableError("injected scan outage")
        return original_scan()

    outbox._entry_paths_sync = flaky_scan
    await outbox.initialize()
    for _ in range(100):
        if scan_attempts:
            break
        await asyncio.sleep(0.01)
    assert scan_attempts == 1

    await outbox.stage(event, media)
    await asyncio.wait_for(publisher.attempted.wait(), timeout=1)
    for _ in range(100):
        if not list(outbox.directory.glob("event-*.json")):
            break
        await asyncio.sleep(0.01)

    assert scan_attempts >= 2
    assert publisher.persisted_ids == {event.id}
    assert not list(outbox.directory.glob("event-*.json"))
    await outbox.close()


async def test_replay_orders_same_camera_events_by_occurrence_time(
    tmp_path,
    sample_event,
) -> None:
    later_source = replace(
        sample_event,
        occurred_at=sample_event.occurred_at + timedelta(seconds=1),
        created_at=sample_event.created_at + timedelta(seconds=1),
    )
    later_event, later_media = _event_and_media(later_source, "evt_later")
    earlier_event, earlier_media = _event_and_media(sample_event, "evt_earlier")
    unavailable = _MediaStorage(fail=True)
    first = _outbox(tmp_path, sample_event, unavailable, _Publisher())
    await first.initialize()
    await first.stage(later_event, later_media)
    await first.stage(earlier_event, earlier_media)
    await first.close()

    publisher = _Publisher()
    restarted = _outbox(tmp_path, sample_event, _MediaStorage(), publisher)
    await restarted.initialize()
    await restarted.replay_once()

    assert publisher.published_ids == [earlier_event.id, later_event.id]
    await restarted.close()


async def test_replay_does_not_overtake_failed_earlier_event(
    tmp_path,
    sample_event,
) -> None:
    later_source = replace(
        sample_event,
        occurred_at=sample_event.occurred_at + timedelta(seconds=1),
        created_at=sample_event.created_at + timedelta(seconds=1),
    )
    later_event, later_media = _event_and_media(later_source, "evt_blocked_later")
    earlier_event, earlier_media = _event_and_media(sample_event, "evt_blocked_earlier")
    first = _outbox(tmp_path, sample_event, _MediaStorage(fail=True), _Publisher())
    await asyncio.to_thread(first._initialize_sync)
    await first.stage(later_event, later_media)
    await first.stage(earlier_event, earlier_media)

    storage = _SelectiveFailMediaStorage(earlier_media[0].key)
    publisher = _Publisher()
    restarted = _outbox(tmp_path, sample_event, storage, publisher)
    await asyncio.to_thread(restarted._initialize_sync)
    await restarted.replay_once()

    assert storage.attempted_keys == [earlier_media[0].key]
    assert publisher.published_ids == []
    assert len(list(restarted.directory.glob("event-*.json"))) == 2


async def test_orphan_temporary_entries_are_removed_before_capacity_check(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_after_orphan")
    outbox = _outbox(
        tmp_path,
        sample_event,
        _MediaStorage(fail=True),
        _Publisher(),
        config=_config(maximum_entries=1),
    )
    await asyncio.to_thread(outbox._initialize_sync)
    orphan = outbox.directory / f".tmp-{uuid.uuid4().hex}"
    orphan.write_bytes(b"abandoned")
    orphan.chmod(0o600)

    await outbox.stage(event, media)

    assert not orphan.exists()
    assert len(list(outbox.directory.glob("event-*.json"))) == 1


async def test_hardlinked_lock_is_rejected_before_mode_change(
    tmp_path,
    sample_event,
) -> None:
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(), _Publisher())
    await asyncio.to_thread(outbox._prepare_root_sync)
    external = tmp_path / "external-lock"
    external.write_bytes(b"")
    external.chmod(0o644)
    os.link(external, outbox.directory / ".lock")

    with pytest.raises(FinalizationOutboxCorruptionError, match="link count"):
        await asyncio.to_thread(outbox._initialize_sync)

    assert stat.S_IMODE(external.stat().st_mode) == 0o644


async def test_hardlinked_entry_is_rejected_before_read(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_hardlinked")
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(fail=True), _Publisher())
    await asyncio.to_thread(outbox._initialize_sync)
    await outbox.stage(event, media)
    [entry] = outbox.directory.glob("event-*.json")
    external = tmp_path / "external-entry"
    os.link(entry, external)

    restarted = _outbox(tmp_path, sample_event, _MediaStorage(), _Publisher())
    with pytest.raises(FinalizationOutboxCorruptionError, match="link count"):
        await asyncio.to_thread(restarted._initialize_sync)

    assert entry.exists()
    assert external.exists()


async def test_close_preserves_caller_cancellation_with_stubborn_worker(
    tmp_path,
    sample_event,
) -> None:
    outbox = _outbox(
        tmp_path,
        sample_event,
        _MediaStorage(),
        _Publisher(),
        config=_config(delivery_timeout_seconds=0.05),
    )
    started = asyncio.Event()
    cancellation_seen = asyncio.Event()
    release = asyncio.Event()

    async def stubborn_worker() -> None:
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release.wait()

    worker = asyncio.create_task(stubborn_worker())
    outbox._task = worker
    await started.wait()
    closing = asyncio.create_task(outbox.close())
    await cancellation_seen.wait()
    closing.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(closing, timeout=0.5)
    assert not worker.done()
    release.set()
    await worker
    outbox._task = None


async def test_initialization_flushes_each_new_directory_parent(
    tmp_path,
    sample_event,
) -> None:
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(), _Publisher())
    flushed = []
    real_fsync = outbox._fsync_directory

    def recording_fsync(path) -> None:
        flushed.append(path)
        real_fsync(path)

    outbox._fsync_directory = recording_fsync
    await outbox.initialize()

    assert tmp_path.resolve() in flushed
    assert outbox.directory.parent in flushed
    assert outbox.directory in flushed
    await outbox.close()


async def test_initialization_flushes_every_new_nested_output_ancestor(
    tmp_path,
    sample_event,
) -> None:
    output = tmp_path / "new-a" / "new-b" / "output"
    outbox = _outbox(output, sample_event, _MediaStorage(), _Publisher())
    flushed = []
    real_fsync = outbox._fsync_directory

    def recording_fsync(path) -> None:
        flushed.append(path)
        real_fsync(path)

    outbox._fsync_directory = recording_fsync
    await outbox.initialize()

    assert {
        tmp_path.resolve(),
        (tmp_path / "new-a").resolve(),
        (tmp_path / "new-a" / "new-b").resolve(),
        output.resolve(),
        outbox.directory.parent,
        outbox.directory,
    }.issubset(set(flushed))
    await outbox.close()


async def test_unexpected_replay_worker_failure_blocks_future_staging(
    tmp_path,
    sample_event,
) -> None:
    event, media = _event_and_media(sample_event, "evt_worker_fatal")
    outbox = _outbox(tmp_path, sample_event, _MediaStorage(), _Publisher())

    def broken_scan():
        raise RuntimeError("injected worker bug")

    outbox._entry_paths_sync = broken_scan
    await outbox.initialize()
    assert outbox._task is not None
    await asyncio.wait_for(outbox._task, timeout=1)

    with pytest.raises(FinalizationOutboxError, match="unexpected failure"):
        await outbox.stage(event, media)
    assert not list(outbox.directory.glob("event-*.json"))
    with pytest.raises(FinalizationOutboxError, match="unexpected failure"):
        await outbox.close()
