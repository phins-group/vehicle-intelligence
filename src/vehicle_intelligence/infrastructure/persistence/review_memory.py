"""Deterministic dataset-sample repository for tests and local development."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime

from vehicle_intelligence.application.ports import DatasetSamplePage, DatasetSampleQuery
from vehicle_intelligence.domain import DatasetSample, DatasetSampleStatus
from vehicle_intelligence.infrastructure.persistence.cursor import decode_cursor, encode_cursor


class InMemoryDatasetSampleRepository:
    def __init__(self) -> None:
        self._samples: dict[str, DatasetSample] = {}
        self._source_revisions: set[tuple[str, int]] = set()
        self._lock = asyncio.Lock()

    async def ensure_indexes(self) -> None:
        return None

    async def create(self, sample: DatasetSample) -> bool:
        source_revision = (sample.source_event_id, sample.review_revision)
        async with self._lock:
            if sample.id in self._samples or source_revision in self._source_revisions:
                return False
            self._samples[sample.id] = sample
            self._source_revisions.add(source_revision)
            return True

    async def get(self, sample_id: str) -> DatasetSample | None:
        return self._samples.get(sample_id)

    async def list(self, query: DatasetSampleQuery) -> DatasetSamplePage:
        items = [sample for sample in self._samples.values() if self._matches(sample, query)]
        items.sort(key=lambda sample: (sample.created_at, sample.id), reverse=True)
        if query.cursor:
            cursor_time, cursor_id = decode_cursor(query.cursor)
            items = [
                sample
                for sample in items
                if (sample.created_at, sample.id) < (cursor_time, cursor_id)
            ]
        page = items[: query.limit + 1]
        has_more = len(page) > query.limit
        page = page[: query.limit]
        next_cursor = encode_cursor(page[-1].created_at, page[-1].id) if has_more and page else None
        return DatasetSamplePage(tuple(page), next_cursor)

    async def claim_for_export(
        self,
        export_id: str,
        limit: int,
        claimed_at: datetime,
        stale_before: datetime,
    ) -> tuple[DatasetSample, ...]:
        if not export_id.strip() or not 1 <= limit <= 1000:
            raise ValueError("dataset export claim is invalid")
        async with self._lock:
            resumed = [
                sample
                for sample in self._samples.values()
                if sample.status is DatasetSampleStatus.EXPORTING and sample.export_id == export_id
            ]
            resumed.sort(key=lambda sample: (sample.created_at, sample.id))
            selected = resumed[:limit]
            selected_ids = {sample.id for sample in selected}
            eligible = [
                sample
                for sample in self._samples.values()
                if sample.id not in selected_ids
                and (
                    sample.status
                    in {
                        DatasetSampleStatus.READY,
                        DatasetSampleStatus.EXPORT_FAILED,
                    }
                    or (
                        sample.status is DatasetSampleStatus.EXPORTING
                        and sample.export_claimed_at is not None
                        and sample.export_claimed_at < stale_before
                    )
                )
            ]
            eligible.sort(key=lambda sample: (sample.created_at, sample.id))
            for sample in eligible[: max(0, limit - len(selected))]:
                claimed = replace(
                    sample,
                    schema_version=max(2, sample.schema_version),
                    status=DatasetSampleStatus.EXPORTING,
                    export_id=export_id,
                    export_attempts=sample.export_attempts + 1,
                    export_claimed_at=claimed_at,
                    exported_at=None,
                    export_manifest_sha256=None,
                    export_error_code=None,
                )
                self._samples[sample.id] = claimed
                selected.append(claimed)
            return tuple(selected)

    async def mark_exported(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        manifest_sha256: str,
        exported_at: datetime,
    ) -> int:
        changed = 0
        async with self._lock:
            for sample_id in dict.fromkeys(sample_ids):
                sample = self._samples.get(sample_id)
                if sample is None:
                    continue
                if (
                    sample.status is DatasetSampleStatus.EXPORTED
                    and sample.export_id == export_id
                    and sample.export_manifest_sha256 == manifest_sha256
                ):
                    changed += 1
                    continue
                if (
                    sample.status is not DatasetSampleStatus.EXPORTING
                    or sample.export_id != export_id
                ):
                    continue
                self._samples[sample_id] = replace(
                    sample,
                    status=DatasetSampleStatus.EXPORTED,
                    exported_at=exported_at,
                    export_manifest_sha256=manifest_sha256,
                    export_error_code=None,
                )
                changed += 1
        return changed

    async def mark_export_failed(
        self,
        sample_ids: tuple[str, ...],
        export_id: str,
        error_code: str,
    ) -> int:
        changed = 0
        async with self._lock:
            for sample_id in dict.fromkeys(sample_ids):
                sample = self._samples.get(sample_id)
                if (
                    sample is None
                    or sample.status is not DatasetSampleStatus.EXPORTING
                    or sample.export_id != export_id
                ):
                    continue
                self._samples[sample_id] = replace(
                    sample,
                    status=DatasetSampleStatus.EXPORT_FAILED,
                    export_error_code=error_code,
                )
                changed += 1
        return changed

    async def close(self) -> None:
        return None

    @staticmethod
    def _matches(sample: DatasetSample, query: DatasetSampleQuery) -> bool:
        return not any(
            (
                query.sample_type is not None and sample.sample_type is not query.sample_type,
                query.status is not None and sample.status is not query.status,
                query.reason is not None and sample.reason is not query.reason,
                query.source_event_id is not None
                and sample.source_event_id != query.source_event_id,
                query.from_time is not None and sample.reviewed_at < query.from_time,
                query.to_time is not None and sample.reviewed_at >= query.to_time,
            )
        )
