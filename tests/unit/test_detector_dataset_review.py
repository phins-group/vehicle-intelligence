from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
from fastapi import FastAPI, Request

from vehicle_intelligence.application.audit import AuditService
from vehicle_intelligence.application.dataset_review import (
    DetectorDatasetReviewService,
    DetectorReviewCommand,
    DetectorReviewQuery,
)
from vehicle_intelligence.application.security import DevelopmentAuthenticator
from vehicle_intelligence.config import AuthConfig, DatasetReviewConfig
from vehicle_intelligence.domain import (
    AuthenticationMethod,
    BoundingBox,
    ModelMetadata,
    PlateDetection,
    Principal,
    UserRole,
)
from vehicle_intelligence.domain.dataset_review import (
    DetectorPromotionStatus,
    DetectorReviewAction,
    DetectorReviewAnnotation,
    DetectorReviewBox,
    DetectorReviewStatus,
)
from vehicle_intelligence.exceptions import (
    DatasetReviewConflictError,
    DatasetReviewStorageError,
    DatasetReviewValidationError,
    PersistenceError,
)
from vehicle_intelligence.infrastructure.persistence.audit_memory import (
    InMemoryAuditLogRepository,
)
from vehicle_intelligence.infrastructure.training.dataset_review_files import (
    FileDetectorReviewRepository,
)
from vehicle_intelligence.interfaces.dataset_review_api import build_dataset_review_router
from vehicle_intelligence.interfaces.request_context import resolve_request_id
from vehicle_intelligence.interfaces.security import APISecurity
from vehicle_intelligence.training.first_party import (
    FirstPartyPlateSourceBuilder,
    verify_first_party_detector_source,
)
from vehicle_intelligence.training.review_suggestions import (
    DetectorReviewSuggestionGenerator,
    ReviewSuggestionModel,
    ReviewSuggestionOptions,
)


class _ReviewSuggestionDetector:
    def detect(self, image: np.ndarray) -> list[PlateDetection]:
        return self._result(image)

    def detect_batch(self, images: list[np.ndarray]) -> list[list[PlateDetection]]:
        return [self._result(image) for image in images]

    @staticmethod
    def _result(image: np.ndarray) -> list[PlateDetection]:
        height, width = image.shape[:2]
        return [
            PlateDetection(
                bbox=BoundingBox(12, 18, min(96, width), min(62, height)),
                confidence=0.88,
                model=ModelMetadata("review-test", "v1", "b" * 64),
            )
        ]


class _FailingAuditRepository(InMemoryAuditLogRepository):
    async def append(self, _entry) -> None:
        raise PersistenceError("audit unavailable")


@pytest.mark.asyncio
async def test_model_suggestions_overlay_unlabeled_items_without_mutating_source(
    tmp_path: Path,
) -> None:
    source_root, source_id = _source(tmp_path)
    source = source_root / source_id
    queue_before = (source / "REVIEW_QUEUE.jsonl").read_bytes()
    manifest_before = (source / "source-manifest.json").read_bytes()
    config = _config(tmp_path, source_root)

    result = DetectorReviewSuggestionGenerator(
        _ReviewSuggestionDetector(),
        ReviewSuggestionModel(
            provider="test",
            name="review-test",
            version="v1",
            sha256="b" * 64,
            confidence=0.7,
            iou=0.6,
            image_size=1280,
        ),
        ReviewSuggestionOptions(
            source_directory=source,
            workspace_directory=config.workspace_directory,
            batch_size=2,
        ),
        clock=lambda: datetime(2026, 8, 11, 4, 0, tzinfo=UTC),
    ).generate()

    assert result.candidates == 1
    assert result.suggested_items == 1
    assert result.suggestion_boxes == 1
    assert result.failures == ()
    assert (source / "REVIEW_QUEUE.jsonl").read_bytes() == queue_before
    assert (source / "source-manifest.json").read_bytes() == manifest_before

    repository = FileDetectorReviewRepository(config)
    service = DetectorDatasetReviewService(repository)
    await service.initialize()
    page = await service.list_items(
        DetectorReviewQuery(
            source_id=source_id,
            reason="MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW",
        )
    )
    generated = next(
        item
        for item in page.items
        if item.suggestions
        and item.suggestions[0].attributes.get("suggestionRunId") == result.suggestion_run_id
    )
    assert generated.suggestions[0].bbox == DetectorReviewBox(12, 18, 84, 44)
    approved = await service.review(
        source_id,
        generated.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    assert approved.status is DetectorReviewStatus.APPROVED
    assert approved.decision is not None
    assert approved.decision.annotations == generated.suggestions
    await service.close()


@pytest.mark.asyncio
async def test_promotion_carries_pending_model_suggestion_overlay_forward(
    tmp_path: Path,
) -> None:
    source_root, source_id = _source(tmp_path)
    config = _config(tmp_path, source_root)
    generated = DetectorReviewSuggestionGenerator(
        _ReviewSuggestionDetector(),
        ReviewSuggestionModel(
            provider="test",
            name="review-test",
            version="v1",
            sha256="b" * 64,
            confidence=0.7,
            iou=0.6,
            image_size=1280,
        ),
        ReviewSuggestionOptions(
            source_directory=source_root / source_id,
            workspace_directory=config.workspace_directory,
        ),
    ).generate()
    repository = FileDetectorReviewRepository(config)
    service = DetectorDatasetReviewService(repository)
    await service.initialize()
    page = await service.list_items(DetectorReviewQuery(source_id=source_id))
    original = next(
        item
        for item in page.items
        if item.suggestions and not item.suggestions[0].attributes.get("suggestionRunId")
    )
    await service.review(
        source_id,
        original.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    job = await repository.create_promotion_job(
        source_id,
        "phins-first-party-suggestion-v2",
        "admin-01",
    )
    await repository.run_promotion_job(job.id)

    target = source_root / "phins-first-party-suggestion-v2"
    manifest, _ = verify_first_party_detector_source(target)
    pending = [
        json.loads(line) for line in (target / "REVIEW_QUEUE.jsonl").read_text().splitlines()
    ]
    assert manifest["reviewQueueCount"] == 1
    assert manifest["statistics"]["modelSuggestionOverlayPendingReview"] == 1
    assert pending[0]["reason"] == "MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW"
    assert pending[0]["suggestions"][0]["attributes"]["suggestionRunId"] == (
        generated.suggestion_run_id
    )
    await service.close()


@pytest.mark.asyncio
async def test_review_overlay_is_revisioned_restart_safe_and_filterable(tmp_path: Path) -> None:
    source_root, source_id = _source(tmp_path)
    config = _config(tmp_path, source_root)
    reviewed_at = datetime(2026, 8, 11, 3, 0, tzinfo=UTC)
    repository = FileDetectorReviewRepository(config, clock=lambda: reviewed_at)
    service = DetectorDatasetReviewService(repository, clock=lambda: reviewed_at)
    await service.initialize()

    summaries = await service.list_sources()
    assert len(summaries) == 1
    assert summaries[0].queue_count == 2
    assert summaries[0].pending_count == 2
    pending = await service.list_items(
        DetectorReviewQuery(source_id=source_id, status=DetectorReviewStatus.PENDING_REVIEW)
    )
    auto = next(item for item in pending.items if item.suggestions)
    unlabeled = next(item for item in pending.items if not item.suggestions)
    detail = await service.get_item(source_id, auto.review_id)
    assert (detail.image_width, detail.image_height) == (140, 100)

    approved = await service.review(
        source_id,
        auto.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    assert approved.status is DetectorReviewStatus.APPROVED
    assert approved.revision == 1
    assert approved.decision is not None
    assert approved.decision.annotations == auto.suggestions

    with pytest.raises(DatasetReviewConflictError, match="expected 0, actual 1"):
        await service.review(
            source_id,
            auto.review_id,
            DetectorReviewCommand(
                action=DetectorReviewAction.APPROVE,
                expected_revision=0,
                reviewer=_operator(),
            ),
        )

    corrected = await service.review(
        source_id,
        unlabeled.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.CORRECT,
            expected_revision=0,
            reviewer=_operator(),
            annotations=(DetectorReviewAnnotation(DetectorReviewBox(18, 22, 76, 31)),),
            note="Manually drawn from the full image",
        ),
    )
    assert corrected.status is DetectorReviewStatus.CORRECTED
    assert (await service.history(source_id, corrected.review_id))[0].note is not None
    assert (
        '"status":"PENDING_REVIEW"' in (source_root / source_id / "REVIEW_QUEUE.jsonl").read_text()
    )

    filtered = await service.list_items(
        DetectorReviewQuery(source_id=source_id, status=DetectorReviewStatus.APPROVED)
    )
    assert [item.review_id for item in filtered.items] == [auto.review_id]
    await service.close()

    restarted = FileDetectorReviewRepository(config, clock=lambda: reviewed_at)
    await restarted.initialize()
    loaded = await restarted.get_item(source_id, auto.review_id)
    assert loaded.status is DetectorReviewStatus.APPROVED
    assert loaded.revision == 1
    await restarted.close()


@pytest.mark.asyncio
async def test_review_validates_actions_and_promotes_new_immutable_source(
    tmp_path: Path,
) -> None:
    source_root, source_id = _source(tmp_path)
    reviewed_at = datetime(2026, 8, 11, 3, 30, tzinfo=UTC)
    config = _config(tmp_path, source_root)
    repository = FileDetectorReviewRepository(
        config,
        clock=lambda: reviewed_at,
    )
    service = DetectorDatasetReviewService(repository, clock=lambda: reviewed_at)
    await service.initialize()
    page = await service.list_items(DetectorReviewQuery(source_id=source_id))
    auto = next(item for item in page.items if item.suggestions)
    unlabeled = next(item for item in page.items if not item.suggestions)

    with pytest.raises(DatasetReviewValidationError, match="at least one human review"):
        await repository.create_promotion_job(
            source_id,
            "phins-first-party-empty-review-v2",
            "admin-01",
        )

    with pytest.raises(DatasetReviewValidationError, match="without model suggestions"):
        await service.review(
            source_id,
            unlabeled.review_id,
            DetectorReviewCommand(
                action=DetectorReviewAction.APPROVE,
                expected_revision=0,
                reviewer=_operator(),
            ),
        )
    with pytest.raises(DatasetReviewValidationError, match="inside the source image"):
        await service.review(
            source_id,
            unlabeled.review_id,
            DetectorReviewCommand(
                action=DetectorReviewAction.CORRECT,
                expected_revision=0,
                reviewer=_operator(),
                annotations=(DetectorReviewAnnotation(DetectorReviewBox(130, 90, 20, 20)),),
            ),
        )

    await service.review(
        source_id,
        auto.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    await service.review(
        source_id,
        unlabeled.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.MARK_NEGATIVE,
            expected_revision=0,
            reviewer=_operator(),
            note="No visible license plate",
        ),
    )
    job = await repository.create_promotion_job(
        source_id,
        "phins-first-party-test-v2",
        "admin-01",
    )
    assert len(job.decision_snapshot_sha256) == 64
    await service.review(
        source_id,
        auto.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.CORRECT,
            expected_revision=1,
            reviewer=_operator(),
            annotations=(DetectorReviewAnnotation(DetectorReviewBox(22, 27, 64, 21)),),
            note="This later revision must not change the queued promotion",
        ),
    )
    await repository.run_promotion_job(job.id)
    completed = await repository.get_promotion_job(job.id)
    assert completed.status is DetectorPromotionStatus.COMPLETED
    assert completed.manifest_sha256 is not None

    target = source_root / "phins-first-party-test-v2"
    manifest, digest = verify_first_party_detector_source(target)
    assert digest == completed.manifest_sha256
    assert manifest["sampleCount"] == 3
    assert manifest["annotationCount"] == 2
    assert manifest["negativeSampleCount"] == 1
    assert manifest["reviewQueueCount"] == 0
    assert manifest["reviewPromotion"]["promotedCount"] == 2
    assert (target / "REVIEW_DECISIONS.jsonl").is_file()
    promoted_evidence = [
        json.loads(line) for line in (target / "REVIEW_DECISIONS.jsonl").read_text().splitlines()
    ]
    promoted_auto = next(item for item in promoted_evidence if item["reviewId"] == auto.review_id)
    assert promoted_auto["revision"] == 1
    assert promoted_auto["status"] == "APPROVED"
    assert {item.source_id for item in await repository.list_sources()} == {
        source_id,
        "phins-first-party-test-v2",
    }
    assert (source_root / source_id / "REVIEW_QUEUE.jsonl").read_text().count("reviewId") == 2
    await service.close()

    restarted = FileDetectorReviewRepository(config, clock=lambda: reviewed_at)
    await restarted.initialize()
    assert (await restarted.get_promotion_job(job.id)).status is DetectorPromotionStatus.COMPLETED
    assert {item.source_id for item in await restarted.list_sources()} == {
        source_id,
        "phins-first-party-test-v2",
    }
    await restarted.close()


@pytest.mark.asyncio
async def test_detector_review_http_contract_serves_authenticated_evidence_and_conflicts(
    tmp_path: Path,
) -> None:
    source_root, source_id = _source(tmp_path)
    repository = FileDetectorReviewRepository(_config(tmp_path, source_root))
    service = DetectorDatasetReviewService(repository)
    audits = AuditService(InMemoryAuditLogRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await audits.initialize()
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_dataset_review_router(service, security, audits))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        sources = await client.get("/api/detector-review/sources")
        assert sources.status_code == 200
        source_summary = sources.json()["items"][0]
        assert source_summary["pendingCount"] == 2
        assert source_summary["sourceType"] == "FIRST_PARTY_DETECTOR_SOURCE"
        assert source_summary["promotionEligible"] is True
        assert source_summary["rightsStatus"] == "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"

        queue = await client.get(
            "/api/detector-review/items",
            params={"sourceId": source_id, "status": "PENDING_REVIEW"},
        )
        auto = next(item for item in queue.json()["items"] if item["suggestions"])
        detail = await client.get(
            f"/api/detector-review/sources/{source_id}/items/{auto['reviewId']}"
        )
        assert detail.status_code == 200
        assert detail.json()["image"] == {"width": 140, "height": 100}
        image = await client.get(
            f"/api/detector-review/sources/{source_id}/items/{auto['reviewId']}/image"
        )
        assert image.status_code == 200
        assert image.headers["content-type"] == "image/jpeg"
        assert image.headers["x-content-type-options"] == "nosniff"

        reviewed = await client.put(
            f"/api/detector-review/sources/{source_id}/items/{auto['reviewId']}",
            json={"action": "APPROVE", "expectedRevision": 0, "annotations": []},
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "APPROVED"
        assert reviewed.headers["x-audit-delivery"] == "delivered"
        stale = await client.put(
            f"/api/detector-review/sources/{source_id}/items/{auto['reviewId']}",
            json={"action": "APPROVE", "expectedRevision": 0, "annotations": []},
        )
        assert stale.status_code == 409
    await audits.close()
    await service.close()


@pytest.mark.asyncio
async def test_detector_review_outbox_recovers_after_audit_outage(tmp_path: Path) -> None:
    source_root, source_id = _source(tmp_path)
    config = _config(tmp_path, source_root)
    repository = FileDetectorReviewRepository(config)
    service = DetectorDatasetReviewService(repository)
    unavailable_audits = AuditService(_FailingAuditRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await unavailable_audits.initialize()
    pending = await service.list_items(DetectorReviewQuery(source_id=source_id))
    suggested = next(item for item in pending.items if item.suggestions)
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_dataset_review_router(service, security, unavailable_audits))
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.put(
            f"/api/detector-review/sources/{source_id}/items/{suggested.review_id}",
            json={"action": "APPROVE", "expectedRevision": 0, "annotations": []},
        )

    assert response.status_code == 200
    assert response.headers["x-audit-delivery"] == "pending"
    queued = await repository.pending_audits()
    assert len(queued) == 1
    decision_path = (
        config.workspace_directory / source_id / "decisions" / suggested.review_id / "00000001.json"
    )
    assert json.loads(decision_path.read_text(encoding="utf-8"))["auditOutbox"]["id"] == (
        queued[0].id
    )
    await unavailable_audits.close()
    await service.close()

    restarted_repository = FileDetectorReviewRepository(config)
    restarted_service = DetectorDatasetReviewService(restarted_repository)
    audit_repository = InMemoryAuditLogRepository()
    recovered_audits = AuditService(audit_repository)
    await restarted_service.initialize()
    await recovered_audits.initialize()
    recovered = await restarted_repository.pending_audits()
    assert recovered == queued

    # Simulate a crash after Mongo accepted the append but before the filesystem
    # delivery marker was created. The retry must recognize the identical audit
    # ID, finish the marker, and never create a second logical record.
    await recovered_audits.persist(queued[0])
    assert await restarted_service.flush_pending_audits(recovered_audits) == 1
    assert await audit_repository.get(queued[0].id) == queued[0]
    assert await restarted_repository.pending_audits() == ()
    await recovered_audits.close()
    await restarted_service.close()

    final_repository = FileDetectorReviewRepository(config)
    await final_repository.initialize()
    assert await final_repository.pending_audits() == ()
    await final_repository.close()


def test_detector_review_audit_marker_rejects_workspace_symlink(tmp_path: Path) -> None:
    source_root, _source_id = _source(tmp_path)
    config = _config(tmp_path, source_root)
    outside = tmp_path / "outside-audit-markers"
    outside.mkdir()
    config.workspace_directory.mkdir(parents=True, exist_ok=True)
    (config.workspace_directory / "audit-outbox").symlink_to(
        outside,
        target_is_directory=True,
    )
    repository = FileDetectorReviewRepository(config)

    with pytest.raises(
        DatasetReviewStorageError,
        match="audit marker directory is unsafe",
    ):
        repository._audit_delivery_path("aud_" + "a" * 32)

    assert list(outside.iterdir()) == []


@pytest.mark.asyncio
async def test_detector_promotion_is_not_dispatched_when_mandatory_audit_fails(
    tmp_path: Path,
) -> None:
    source_root, source_id = _source(tmp_path)
    config = _config(tmp_path, source_root)
    repository = FileDetectorReviewRepository(config)
    service = DetectorDatasetReviewService(repository)
    audits = AuditService(_FailingAuditRepository())
    security = APISecurity(AuthConfig(enabled=False), DevelopmentAuthenticator())
    await service.initialize()
    await audits.initialize()
    pending = await service.list_items(DetectorReviewQuery(source_id=source_id))
    suggested = next(item for item in pending.items if item.suggestions)
    await service.review(
        source_id,
        suggested.review_id,
        DetectorReviewCommand(
            action=DetectorReviewAction.APPROVE,
            expected_revision=0,
            reviewer=_operator(),
        ),
    )
    app = FastAPI()

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request.state.request_id = resolve_request_id(request)
        return await call_next(request)

    app.include_router(build_dataset_review_router(service, security, audits))
    transport = httpx.ASGITransport(app=app)
    target_source_id = "phins-first-party-audit-v2"
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            f"/api/detector-review/sources/{source_id}/promotions",
            json={"targetSourceId": target_source_id},
        )

    assert response.status_code == 503
    job_paths = sorted((config.workspace_directory / "promotion-jobs").glob("promotion-*.json"))
    job_paths = [path for path in job_paths if not path.name.endswith(".decisions.json")]
    assert len(job_paths) == 1
    job = json.loads(job_paths[0].read_text(encoding="utf-8"))
    assert job["status"] == "FAILED"
    assert job["errorCode"] == "AUDIT_WRITE_FAILED"
    assert not (source_root / target_source_id).exists()
    await audits.close()
    await service.close()


def _source(tmp_path: Path) -> tuple[Path, str]:
    labels = tmp_path / "labels"
    (labels / "images").mkdir(parents=True)
    labeled = _jpg(80, plate=True)
    (labels / "images/labeled.jpg").write_bytes(labeled)
    (labels / "annotations.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "reference",
                "imagePath": "images/labeled.jpg",
                "groupId": "reference-group",
                "cameraId": "reference-camera",
                "capturedAt": "2026-08-01T00:00:00Z",
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 20, "y": 25, "width": 70, "height": 25},
                    }
                ],
            }
        )
        + "\n"
    )
    auto = tmp_path / "auto"
    (auto / "images").mkdir(parents=True)
    auto_image = _jpg(110, plate=True)
    (auto / "images/auto.jpg").write_bytes(auto_image)
    (auto / "annotations.auto.jsonl").write_text(
        json.dumps(
            {
                "sampleId": "auto",
                "imagePath": "images/auto.jpg",
                "groupId": "auto-group",
                "cameraId": "auto-camera",
                "capturedAt": "2026-08-02T00:00:00Z",
                "annotations": [
                    {
                        "className": "license_plate",
                        "bbox": {"x": 20, "y": 25, "width": 70, "height": 25},
                        "attributes": {"confidence": 0.91},
                    }
                ],
            }
        )
        + "\n"
    )
    collected = tmp_path / "collected"
    collected.mkdir()
    (collected / "labeled.jpg").write_bytes(labeled)
    (collected / "auto.jpg").write_bytes(auto_image)
    (collected / "unlabeled.jpg").write_bytes(_jpg(140, plate=False))
    sources = tmp_path / "sources"
    source_id = "phins-first-party-test-v1"
    FirstPartyPlateSourceBuilder(
        input_directory=collected,
        output_directory=sources / source_id,
        label_reference_directory=labels,
        auto_reference_directory=auto,
        source_id=source_id,
        owner_namespace="phins-group",
        founder_id="duyhuynh",
    ).build()
    return sources, source_id


def _config(tmp_path: Path, sources: Path) -> DatasetReviewConfig:
    return DatasetReviewConfig(
        sources_directory=sources,
        workspace_directory=tmp_path / "review-workspace",
        promoted_sources_directory=sources,
    )


def _operator() -> Principal:
    return Principal(
        id="operator-01",
        display_name="Dataset Operator",
        role=UserRole.OPERATOR,
        authentication_method=AuthenticationMethod.DEVELOPMENT,
    )


def _jpg(fill: int, *, plate: bool) -> bytes:
    image = np.full((100, 140, 3), fill, dtype=np.uint8)
    if plate:
        cv2.rectangle(image, (20, 25), (90, 50), (245, 245, 245), -1)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()
