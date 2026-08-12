"""Application use cases for immutable dataset catalog and private Hub sync."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from vehicle_intelligence.domain import Principal
from vehicle_intelligence.domain.dataset_registry import (
    DatasetHubSyncJob,
    DatasetHubSyncStatus,
    DatasetRegistryCapabilities,
    DetectorDatasetSampleImage,
    DetectorDatasetSampleKind,
    DetectorDatasetSamplePage,
    DetectorDatasetVersion,
)


@dataclass(frozen=True, slots=True)
class DatasetHubSyncCommand:
    export_id: str
    revision: str = "main"
    restricted_transfer_confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DetectorDatasetSampleQuery:
    source_id: str
    limit: int = 12
    cursor: str | None = None
    kind: DetectorDatasetSampleKind = DetectorDatasetSampleKind.ALL
    lighting: str | None = None


class DatasetRegistryRepository(Protocol):
    async def initialize(self) -> None: ...

    async def close(self) -> None: ...

    async def capabilities(self) -> DatasetRegistryCapabilities: ...

    async def list_datasets(self) -> tuple[DetectorDatasetVersion, ...]: ...

    async def list_samples(
        self,
        query: DetectorDatasetSampleQuery,
    ) -> DetectorDatasetSamplePage: ...

    async def get_sample_image(
        self,
        source_id: str,
        image_sha256: str,
    ) -> DetectorDatasetSampleImage: ...

    async def create_sync_job(
        self,
        source_id: str,
        command: DatasetHubSyncCommand,
        requested_by: str,
    ) -> DatasetHubSyncJob: ...

    async def run_sync_job(self, job_id: str) -> None: ...

    async def get_sync_job(self, job_id: str) -> DatasetHubSyncJob: ...

    async def fail_queued_sync_job(self, job_id: str, error_code: str) -> None: ...


class DatasetRegistryService:
    def __init__(self, repository: DatasetRegistryRepository) -> None:
        self._repository = repository
        self._tasks: set[asyncio.Task[None]] = set()

    async def initialize(self) -> None:
        await self._repository.initialize()

    async def close(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._repository.close()

    async def capabilities(self) -> DatasetRegistryCapabilities:
        return await self._repository.capabilities()

    async def list_datasets(self) -> tuple[DetectorDatasetVersion, ...]:
        return await self._repository.list_datasets()

    async def list_samples(
        self,
        query: DetectorDatasetSampleQuery,
    ) -> DetectorDatasetSamplePage:
        return await self._repository.list_samples(query)

    async def get_sample_image(
        self,
        source_id: str,
        image_sha256: str,
    ) -> DetectorDatasetSampleImage:
        return await self._repository.get_sample_image(source_id, image_sha256)

    async def start_sync(
        self,
        source_id: str,
        command: DatasetHubSyncCommand,
        principal: Principal,
    ) -> DatasetHubSyncJob:
        job = await self.prepare_sync(source_id, command, principal)
        self.dispatch_sync(job)
        return job

    async def prepare_sync(
        self,
        source_id: str,
        command: DatasetHubSyncCommand,
        principal: Principal,
    ) -> DatasetHubSyncJob:
        """Persist a validated job without starting its external side effect."""

        return await self._repository.create_sync_job(source_id, command, principal.id)

    def dispatch_sync(self, job: DatasetHubSyncJob) -> None:
        """Start a queued job after its audit record has been durably written."""

        if job.status is DatasetHubSyncStatus.QUEUED and not any(
            task.get_name() == f"dataset-hub-sync-{job.id}" for task in self._tasks
        ):
            task = asyncio.create_task(
                self._repository.run_sync_job(job.id),
                name=f"dataset-hub-sync-{job.id}",
            )
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def fail_prepared_sync(self, job_id: str, error_code: str) -> None:
        await self._repository.fail_queued_sync_job(job_id, error_code)

    async def get_sync(self, job_id: str) -> DatasetHubSyncJob:
        return await self._repository.get_sync_job(job_id)
