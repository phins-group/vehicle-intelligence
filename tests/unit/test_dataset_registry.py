from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.dataset_registry import (
    DatasetHubSyncCommand,
    DatasetRegistryService,
    DetectorDatasetSampleQuery,
)
from vehicle_intelligence.application.security import DevelopmentAuthenticator
from vehicle_intelligence.config import AuthConfig, DatasetRegistryConfig
from vehicle_intelligence.domain.dataset_registry import (
    DatasetHubSyncStatus,
    DetectorDatasetSampleKind,
)
from vehicle_intelligence.exceptions import (
    DatasetRegistryValidationError,
    InvalidCursorError,
    PersistenceError,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.training.dataset_registry_files import (
    FileDatasetRegistryRepository,
)
from vehicle_intelligence.interfaces.dataset_registry_api import build_dataset_registry_router
from vehicle_intelligence.interfaces.request_context import resolve_request_id
from vehicle_intelligence.interfaces.security import APISecurity
from vehicle_intelligence.training.config import DetectorDatasetConfig, SplitConfig
from vehicle_intelligence.training.domain import DetectorRole, HubUploadResult
from vehicle_intelligence.training.first_party import FirstPartyPlateSourceBuilder


class _FakeUploader:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def upload_dataset(
        self,
        directory: Path,
        repo_id: str,
        *,
        revision: str = "main",
        allow_restricted_private: bool = False,
    ) -> HubUploadResult:
        self.calls.append(
            {
                "directory": directory,
                "repo_id": repo_id,
                "revision": revision,
                "allow_restricted_private": allow_restricted_private,
            }
        )
        return HubUploadResult(
            repo_id=repo_id,
            repo_type="dataset",
            revision="commit123",
            url="https://huggingface.co/datasets/phins-group/plate-dataset/commit/commit123",
        )


class _FailingAuditRepository(InMemoryAuditLogRepository):
    async def append(self, _entry) -> None:
        raise PersistenceError("audit unavailable")


@pytest.mark.asyncio
async def test_registry_builds_verified_export_and_idempotently_syncs_private_source(
    tmp_path: Path,
) -> None:
    repository, uploader, source_id = _repository(tmp_path)
    await repository.initialize()

    datasets = await repository.list_datasets()
    assert [item.source_id for item in datasets] == [source_id]
    assert datasets[0].review_queue_count == 0
    assert datasets[0].release_eligible is True
    assert datasets[0].distribution_eligible is False

    with pytest.raises(DatasetRegistryValidationError, match="explicitly confirmed"):
        await repository.create_sync_job(
            source_id,
            DatasetHubSyncCommand(export_id="plate-production-v2"),
            "admin-01",
        )

    command = DatasetHubSyncCommand(
        export_id="plate-production-v2",
        revision="main",
        restricted_transfer_confirmed=True,
    )
    queued = await repository.create_sync_job(source_id, command, "admin-01")
    assert queued.status is DatasetHubSyncStatus.QUEUED

    await repository.run_sync_job(queued.id)
    completed = await repository.get_sync_job(queued.id)
    assert completed.status is DatasetHubSyncStatus.COMPLETED
    assert completed.hub_commit_sha == "commit123"
    assert completed.export_manifest_sha256 is not None
    assert completed.reused_export is False
    assert uploader.calls == [
        {
            "directory": tmp_path / "exports" / "plate-production-v2",
            "repo_id": "phins-group/plate-dataset",
            "revision": "main",
            "allow_restricted_private": True,
        }
    ]

    catalog = await repository.list_datasets()
    assert catalog[0].export is not None
    assert catalog[0].export.source_manifest_sha256 == catalog[0].source_manifest_sha256
    assert catalog[0].latest_sync == completed
    assert await repository.create_sync_job(source_id, command, "admin-02") == completed
    await repository.close()

    restarted = FileDatasetRegistryRepository(
        _registry_config(tmp_path),
        dataset_config=_dataset_config(tmp_path, source_id),
        hub_repo_id="phins-group/plate-dataset",
        hub_enabled=True,
        uploader=uploader,
    )
    await restarted.initialize()
    assert (await restarted.get_sync_job(completed.id)).status is DatasetHubSyncStatus.COMPLETED
    await restarted.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_type",
    ["VIDEO_DETECTOR_REVIEW_SOURCE", "WAREHOUSE_PLATE_REVIEW_SOURCE"],
)
async def test_registry_ignores_review_only_sources(
    tmp_path: Path,
    source_type: str,
) -> None:
    _ready_source(tmp_path)
    review_id = "phins-video-review-only-v1"
    review = tmp_path / "sources" / review_id
    review.mkdir()
    (review / "source-manifest.json").write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "type": source_type,
                "role": "plate",
                "sourceId": review_id,
                "releaseEligible": False,
                "distributionEligible": False,
                "promotionEligible": False,
            }
        ),
        encoding="utf-8",
    )
    repository = FileDatasetRegistryRepository(
        _registry_config(tmp_path),
        dataset_config=_dataset_config(tmp_path, "phins-first-party-ready-v2"),
        hub_repo_id="phins-group/plate-dataset",
        hub_enabled=True,
        uploader=_FakeUploader(),
    )
    await repository.initialize()
    assert [item.source_id for item in await repository.list_datasets()] == [
        "phins-first-party-ready-v2"
    ]
    await repository.close()


@pytest.mark.asyncio
async def test_dataset_registry_http_contract_requires_confirmation_and_tracks_job(
    tmp_path: Path,
) -> None:
    repository, uploader, source_id = _repository(tmp_path)
    service = DatasetRegistryService(repository)
    audits = AuditService(InMemoryAuditLogRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await audits.initialize()
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_dataset_registry_router(service, security, audits))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        catalog = await client.get("/api/datasets")
        assert catalog.status_code == 200
        assert catalog.headers["cache-control"] == "no-store, private"
        assert catalog.json()["hub"] == {
            "enabled": True,
            "hubEnabled": True,
            "repoId": "phins-group/plate-dataset",
            "credentialsConfigured": True,
            "restrictedPrivateSyncEnabled": True,
        }

        samples = await client.get(f"/api/datasets/{source_id}/samples?limit=2")
        assert samples.status_code == 200
        assert samples.headers["cache-control"] == "no-store, private"
        sample_page = samples.json()
        assert len(sample_page["items"]) == 2
        assert sample_page["nextCursor"]
        assert sample_page["items"][0]["image"] == {"width": 140, "height": 100}
        assert sample_page["items"][0]["annotations"][0]["className"] == "license_plate"

        image = await client.get(sample_page["items"][0]["imageUrl"])
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.headers["cache-control"] == "private, max-age=300"
        assert image.headers["x-content-type-options"] == "nosniff"
        assert image.content.startswith(b"\xff\xd8")

        negatives = await client.get(
            f"/api/datasets/{source_id}/samples",
            params={"kind": "NEGATIVE"},
        )
        assert negatives.status_code == 200
        assert len(negatives.json()["items"]) == 1
        assert negatives.json()["items"][0]["negative"] is True
        assert negatives.json()["items"][0]["annotations"] == []

        refused = await client.post(
            f"/api/datasets/{source_id}/syncs",
            json={"exportId": "plate-api-v2", "revision": "main"},
        )
        assert refused.status_code == 422

        accepted = await client.post(
            f"/api/datasets/{source_id}/syncs",
            json={
                "exportId": "plate-api-v2",
                "revision": "main",
                "confirmRestrictedPrivateTransfer": True,
            },
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["id"]

        status = accepted.json()
        for _ in range(100):
            status_response = await client.get(f"/api/datasets/syncs/{job_id}")
            assert status_response.status_code == 200
            status = status_response.json()
            if status["status"] not in {"QUEUED", "PREPARING_EXPORT", "UPLOADING"}:
                break
            await asyncio.sleep(0.01)
        assert status["status"] == "COMPLETED"
        assert status["hubCommitSha"] == "commit123"
        assert status["restrictedTransferConfirmed"] is True
        assert len(uploader.calls) == 1

    await service.close()
    await audits.close()


@pytest.mark.asyncio
async def test_dataset_registry_pages_and_filters_checksum_verified_samples(
    tmp_path: Path,
) -> None:
    repository, _uploader, source_id = _repository(tmp_path)
    await repository.initialize()

    first = await repository.list_samples(DetectorDatasetSampleQuery(source_id=source_id, limit=2))
    assert len(first.items) == 2
    assert first.next_cursor is not None
    assert all(item.image_width == 140 and item.image_height == 100 for item in first.items)

    second = await repository.list_samples(
        DetectorDatasetSampleQuery(
            source_id=source_id,
            limit=2,
            cursor=first.next_cursor,
        )
    )
    assert len(second.items) == 1
    assert second.next_cursor is None
    assert sum(item.negative for item in (*first.items, *second.items)) == 1

    positive = await repository.list_samples(
        DetectorDatasetSampleQuery(
            source_id=source_id,
            kind=DetectorDatasetSampleKind.POSITIVE,
        )
    )
    assert len(positive.items) == 2
    assert all(not item.negative and len(item.annotations) == 1 for item in positive.items)

    negative = await repository.list_samples(
        DetectorDatasetSampleQuery(
            source_id=source_id,
            kind=DetectorDatasetSampleKind.NEGATIVE,
        )
    )
    assert len(negative.items) == 1
    assert negative.items[0].negative is True
    assert negative.items[0].annotations == ()

    image = await repository.get_sample_image(
        source_id,
        positive.items[0].image_sha256,
    )
    assert Path(image.path).read_bytes().startswith(b"\xff\xd8")
    assert image.media_type == "image/jpeg"
    assert image.sha256 == positive.items[0].image_sha256

    with pytest.raises(InvalidCursorError):
        await repository.list_samples(
            DetectorDatasetSampleQuery(
                source_id=source_id,
                cursor=first.next_cursor,
                kind=DetectorDatasetSampleKind.NEGATIVE,
            )
        )

    await repository.close()


@pytest.mark.asyncio
async def test_dataset_sync_is_not_dispatched_when_mandatory_audit_fails(
    tmp_path: Path,
) -> None:
    repository, uploader, source_id = _repository(tmp_path)
    service = DatasetRegistryService(repository)
    audits = AuditService(_FailingAuditRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await audits.initialize()
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_dataset_registry_router(service, security, audits))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/datasets/{source_id}/syncs",
            json={
                "exportId": "plate-audit-v2",
                "revision": "main",
                "confirmRestrictedPrivateTransfer": True,
            },
        )
        assert response.status_code == 503
        await asyncio.sleep(0)
        assert uploader.calls == []
        catalog = await client.get("/api/datasets")
        assert catalog.json()["items"][0]["latestSync"]["status"] == "FAILED"
        assert catalog.json()["items"][0]["latestSync"]["errorCode"] == "AUDIT_WRITE_FAILED"

    await service.close()
    await audits.close()


def _repository(tmp_path: Path) -> tuple[FileDatasetRegistryRepository, _FakeUploader, str]:
    source_id = _ready_source(tmp_path)
    uploader = _FakeUploader()
    repository = FileDatasetRegistryRepository(
        _registry_config(tmp_path),
        dataset_config=_dataset_config(tmp_path, source_id),
        hub_repo_id="phins-group/plate-dataset",
        hub_enabled=True,
        uploader=uploader,
    )
    return repository, uploader, source_id


def _registry_config(tmp_path: Path) -> DatasetRegistryConfig:
    return DatasetRegistryConfig(
        sources_directory=tmp_path / "sources",
        exports_directory=tmp_path / "exports",
        workspace_directory=tmp_path / "registry",
        training_config=tmp_path / "unused-training-config.yaml",
        restricted_private_sync_enabled=True,
    )


def _dataset_config(tmp_path: Path, source_id: str) -> DetectorDatasetConfig:
    return DetectorDatasetConfig(
        role=DetectorRole.PLATE,
        source_directory=tmp_path / "sources" / source_id,
        output_directory=tmp_path / "exports",
        classes=("license_plate",),
        split=SplitConfig(require_non_empty=False),
    )


def _ready_source(tmp_path: Path) -> str:
    labels = tmp_path / "labels"
    label_images = labels / "images"
    label_images.mkdir(parents=True)
    collected = tmp_path / "collected"
    collected.mkdir()
    records: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        filename = f"plate-{index}.jpg"
        image = _jpg(60 + index * 30, index)
        (label_images / filename).write_bytes(image)
        (collected / filename).write_bytes(image)
        annotations = (
            []
            if split == "test"
            else [
                {
                    "className": "license_plate",
                    "bbox": {"x": 20, "y": 25, "width": 70, "height": 25},
                }
            ]
        )
        records.append(
            {
                "sampleId": f"reference-{index}",
                "imagePath": f"images/{filename}",
                "groupId": f"sequence-{index}",
                "cameraId": "warehouse-gate",
                "capturedAt": f"2026-08-{index:02d}T00:00:00Z",
                "split": split,
                "annotations": annotations,
            }
        )
    (labels / "annotations.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    source_id = "phins-first-party-ready-v2"
    FirstPartyPlateSourceBuilder(
        input_directory=collected,
        output_directory=tmp_path / "sources" / source_id,
        label_reference_directory=labels,
        source_id=source_id,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    return source_id


def _jpg(fill: int, marker: int) -> bytes:
    image = np.full((100, 140, 3), fill, dtype=np.uint8)
    image[marker, marker] = (fill + 1, fill + 2, fill + 3)
    cv2.rectangle(image, (20, 25), (90, 50), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
