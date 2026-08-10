import asyncio
from dataclasses import replace
from datetime import timedelta

from fastapi.testclient import TestClient

from vehicle_intelligence.application.identity import VehicleIdentityProcessor, bootstrap_vehicle_id
from vehicle_intelligence.application.media_access import VehicleEventMediaService
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.review import HumanPlateReviewService
from vehicle_intelligence.application.topology import CameraTopologyService, TopologyCreate
from vehicle_intelligence.config import IdentityConfig, load_settings
from vehicle_intelligence.domain import (
    CameraSnapshot,
    EventStatus,
    VehicleFingerprint,
    VehicleIdentity,
)
from vehicle_intelligence.infrastructure.persistence.identity_memory import (
    InMemoryVectorRepository,
    InMemoryVehicleIdentityRepository,
)
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)
from vehicle_intelligence.infrastructure.persistence.topology_memory import (
    InMemoryCameraTopologyRepository,
)
from vehicle_intelligence.interfaces.api import create_app


class FakeMediaUrlSigner:
    def __init__(self) -> None:
        self.keys: list[str] = []

    async def presign_get(self, key: str, expires: timedelta) -> str | None:
        self.keys.append(key)
        return f"https://media.example/{key}?signature=test"


def test_api_lists_and_searches_normalized_plate(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    app = create_app(load_settings(), repository)
    asyncio.run(repository.save(sample_event))

    with TestClient(app) as client:
        health = client.get("/api/system/health")
        events = client.get("/api/events", params={"plate": "51H12345"})
        search = client.get("/api/vehicles/search", params={"plate": "51H 12345"})
        detail = client.get(f"/api/events/{sample_event.id}")

    assert health.status_code == 200
    assert health.json()["phase"] == "4"
    assert events.status_code == 200
    assert events.json()["items"][0]["_id"] == sample_event.id
    assert search.json()["query"] == "51H-123.45"
    assert search.json()["nextCursor"] is None
    assert detail.json()["plate"]["normalized"] == "51H-123.45"


def test_api_exposes_logical_identity_and_bounded_fingerprints(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository()
    asyncio.run(events.save(sample_event))
    processor = VehicleIdentityProcessor(identities, events, IdentityConfig())
    asyncio.run(processor.process(sample_event))
    vehicle_id = bootstrap_vehicle_id(sample_event.id)
    later = replace(
        sample_event,
        id="evt_identity_later",
        track_id="warehouse:identity:2",
        camera=CameraSnapshot("warehouse", "Warehouse", "ZONE_B"),
        occurred_at=sample_event.occurred_at + timedelta(minutes=4),
        created_at=sample_event.created_at + timedelta(minutes=4),
        vehicle_id=vehicle_id,
    )
    asyncio.run(events.save(later))
    app = create_app(
        load_settings(),
        events,
        vehicle_identity_repository=identities,
    )

    with TestClient(app) as client:
        detail = client.get(f"/api/vehicles/{vehicle_id}")
        fingerprints = client.get(f"/api/vehicles/{vehicle_id}/fingerprints")
        timeline = client.get(f"/api/vehicles/{vehicle_id}/timeline")
        journey = client.get(f"/api/vehicles/{vehicle_id}/journey")
        missing = client.get("/api/vehicles/veh_missing")

    assert detail.status_code == 200
    assert detail.json()["primaryPlate"] == "51H-123.45"
    assert detail.json()["observationCount"] == 1
    assert detail.json()["latestEvent"]["camera"]["id"] == "warehouse"
    assert fingerprints.status_code == 200
    assert fingerprints.json()["items"][0]["sourceEventId"] == sample_event.id
    assert fingerprints.json()["items"][0]["embedding"] is None
    assert [item["cameraId"] for item in timeline.json()["items"]] == [
        "gate-01",
        "warehouse",
    ]
    assert journey.json()["segments"][0]["feasible"] is None
    assert missing.status_code == 404


def test_api_manages_topology_and_returns_time_bounded_candidates(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository()
    topology = InMemoryCameraTopologyRepository()
    now = sample_event.occurred_at
    observations = (
        (
            VehicleIdentity(
                "veh-current",
                None,
                (),
                "car",
                None,
                now,
                now,
                1,
            ),
            VehicleFingerprint(
                "fp-current",
                "veh-current",
                "evt-current",
                "camera-b",
                now,
                "car",
                0.9,
            ),
        ),
        (
            VehicleIdentity(
                "veh-prior",
                None,
                (),
                "car",
                None,
                now - timedelta(seconds=300),
                now - timedelta(seconds=300),
                1,
            ),
            VehicleFingerprint(
                "fp-prior",
                "veh-prior",
                "evt-prior",
                "camera-a",
                now - timedelta(seconds=300),
                "car",
                0.9,
            ),
        ),
    )
    for identity, fingerprint in observations:
        asyncio.run(identities.register_observation(identity, fingerprint))
    app = create_app(
        load_settings(),
        events,
        vehicle_identity_repository=identities,
        topology_repository=topology,
    )
    camera_payload = {
        "name": "Topology Camera",
        "stream": {"rtspUrl": "rtsp://camera.example/live", "fpsLimit": 6},
    }
    edge_payload = {
        "id": "camera-a-to-b",
        "fromCameraId": "camera-a",
        "toCameraId": "camera-b",
        "travelTime": {
            "minimumSeconds": 60,
            "maximumSeconds": 600,
            "typicalSeconds": 300,
        },
        "enabled": True,
    }

    with TestClient(app) as client:
        for camera_id in ("camera-a", "camera-b"):
            response = client.post(
                "/api/cameras",
                json={"id": camera_id, **camera_payload},
            )
            assert response.status_code == 201
        created = client.post("/api/camera-topology", json=edge_payload)
        listed = client.get("/api/camera-topology", params={"toCameraId": "camera-b"})
        candidates = client.get("/api/vehicle-fingerprints/fp-current/candidates")
        stale = client.put(
            "/api/camera-topology/camera-a-to-b",
            json={
                "revision": 99,
                **{key: value for key, value in edge_payload.items() if key != "id"},
            },
        )
        removed = client.delete("/api/camera-topology/camera-a-to-b")

    assert created.status_code == 201
    assert listed.json()["items"][0]["fromCameraId"] == "camera-a"
    assert candidates.status_code == 200
    assert candidates.json()["items"][0]["fingerprintId"] == "fp-prior"
    assert candidates.json()["items"][0]["timeScore"] == 1
    assert stale.status_code == 409
    assert removed.status_code == 204


def test_api_scores_and_reviews_identity_merge_split(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    identities = InMemoryVehicleIdentityRepository(events)
    topology = InMemoryCameraTopologyRepository()
    vectors = InMemoryVectorRepository()
    now = sample_event.occurred_at
    asyncio.run(
        CameraTopologyService(topology, clock=lambda: now).create(
            TopologyCreate("edge-a-b", "camera-a", "camera-b", 60, 600, 300)
        )
    )
    source_event = replace(
        sample_event,
        id="evt-api-reid-source",
        track_id="camera-b:api-reid-source",
        vehicle_id="veh-api-source",
    )
    target_event = replace(
        sample_event,
        id="evt-api-reid-target",
        track_id="camera-a:api-reid-target",
        occurred_at=now - timedelta(seconds=300),
        created_at=now - timedelta(seconds=300),
        vehicle_id="veh-api-target",
    )
    asyncio.run(events.save(source_event))
    asyncio.run(events.save(target_event))
    for identity, fingerprint in (
        (
            VehicleIdentity("veh-api-source", None, (), "car", "white", now, now, 1),
            VehicleFingerprint(
                "fp-api-source",
                "veh-api-source",
                source_event.id,
                "camera-b",
                now,
                "car",
                0.95,
                "51H-123.45",
                0.95,
                "white",
            ),
        ),
        (
            VehicleIdentity(
                "veh-api-target",
                None,
                (),
                "car",
                "white",
                now - timedelta(seconds=300),
                now - timedelta(seconds=300),
                1,
            ),
            VehicleFingerprint(
                "fp-api-target",
                "veh-api-target",
                target_event.id,
                "camera-a",
                now - timedelta(seconds=300),
                "car",
                0.95,
                "51H-123.45",
                0.95,
                "white",
            ),
        ),
    ):
        asyncio.run(identities.register_observation(identity, fingerprint))
    app = create_app(
        load_settings(),
        events,
        vehicle_identity_repository=identities,
        topology_repository=topology,
        vector_repository=vectors,
    )
    merge_payload = {
        "reviewId": "api-merge-001",
        "sourceVehicleId": "veh-api-source",
        "targetVehicleId": "veh-api-target",
        "expectedSourceRevision": 1,
        "expectedTargetRevision": 1,
        "reason": "Operator confirmed matching body damage",
        "sourceFingerprintId": "fp-api-source",
        "targetFingerprintId": "fp-api-target",
    }

    with TestClient(app) as client:
        scored = client.get("/api/vehicle-fingerprints/fp-api-source/reid-candidates")
        merged = client.post("/api/vehicle-identities/merge", json=merge_payload)
        merge_retry = client.post("/api/vehicle-identities/merge", json=merge_payload)
        split = client.post(
            "/api/vehicle-identities/split",
            json={
                "reviewId": "api-split-001",
                "sourceVehicleId": "veh-api-target",
                "expectedSourceRevision": 2,
                "fingerprintIds": ["fp-api-source"],
                "reason": "Second reviewer found a distinguishing feature",
            },
        )
        review = client.get("/api/vehicle-identity-reviews/api-split-001")
        audits = client.get(
            "/api/audit-logs",
            params={"resourceType": "VEHICLE_IDENTITY"},
        )

    assert scored.status_code == 200
    assert scored.json()["items"][0]["verdict"] == "MATCH"
    assert merged.status_code == 200 and merged.json()["movedEvents"] == 1
    assert merge_retry.json()["idempotent"] is True
    assert split.status_code == 200 and split.json()["movedEvents"] == 1
    assert review.json()["action"] == "SPLIT"
    assert len(audits.json()["items"]) == 2


def test_vehicle_search_uses_stable_cursor_pagination(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    older = replace(
        sample_event,
        id="evt_vehicle_search_old",
        track_id="gate-01:video-test:10",
        occurred_at=sample_event.occurred_at - timedelta(minutes=2),
        created_at=sample_event.created_at - timedelta(minutes=2),
    )
    newer = replace(
        sample_event,
        id="evt_vehicle_search_new",
        track_id="gate-01:video-test:11",
        occurred_at=sample_event.occurred_at - timedelta(minutes=1),
        created_at=sample_event.created_at - timedelta(minutes=1),
    )
    asyncio.run(repository.save(older))
    asyncio.run(repository.save(newer))
    app = create_app(load_settings(), repository)

    with TestClient(app) as client:
        first = client.get(
            "/api/vehicles/search",
            params={"plate": "51H12345", "limit": 1},
        )
        second = client.get(
            "/api/vehicles/search",
            params={
                "plate": "51H12345",
                "limit": 1,
                "cursor": first.json()["nextCursor"],
            },
        )
        invalid = client.get(
            "/api/vehicles/search",
            params={"plate": "51H12345", "cursor": "invalid-cursor"},
        )

    assert first.status_code == 200
    assert first.json()["items"][0]["_id"] == newer.id
    assert first.json()["nextCursor"] is not None
    assert second.status_code == 200
    assert second.json()["items"][0]["_id"] == older.id
    assert second.json()["nextCursor"] is None
    assert invalid.status_code == 400


def test_event_media_access_is_event_scoped_short_lived_and_not_cached(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    asyncio.run(repository.save(sample_event))
    signer = FakeMediaUrlSigner()
    now = sample_event.occurred_at
    service = VehicleEventMediaService(repository, signer, 120, clock=lambda: now)
    app = create_app(load_settings(), repository, media_access_service=service)

    with TestClient(app) as client:
        media = client.get(f"/api/events/{sample_event.id}/media")
        missing_event = client.get("/api/events/evt-missing/media")

    assert media.status_code == 200
    assert media.headers["cache-control"] == "no-store, private"
    assert media.headers["pragma"] == "no-cache"
    assert media.json() == {
        "eventId": sample_event.id,
        "expiresAt": (now + timedelta(seconds=120)).isoformat().replace("+00:00", "Z"),
        "media": {
            "snapshot": {
                "key": "vehicles/test/snapshot.jpg",
                "url": "https://media.example/vehicles/test/snapshot.jpg?signature=test",
                "contentType": "image/jpeg",
                "status": "AVAILABLE",
            },
            "vehicleCrop": None,
            "plateCrop": None,
            "clip": None,
        },
    }
    assert signer.keys == ["vehicles/test/snapshot.jpg"]
    assert missing_event.status_code == 404


def test_event_media_reports_unconfigured_local_storage(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    asyncio.run(repository.save(sample_event))
    app = create_app(load_settings(), repository)

    with TestClient(app) as client:
        response = client.get(f"/api/events/{sample_event.id}/media")

    assert response.status_code == 503
    assert response.json()["detail"] == "media access requires configured object storage"


def test_api_reviews_plate_audits_change_and_exposes_dataset_feedback(sample_event) -> None:
    repository = InMemoryVehicleEventRepository()
    review_event = replace(
        sample_event,
        status=EventStatus.NEEDS_REVIEW,
        media=replace(sample_event.media, plate_crop_key="vehicles/test/plate.jpg"),
    )
    asyncio.run(repository.save(review_event))
    app = create_app(
        load_settings(),
        repository,
        human_review_service=HumanPlateReviewService(
            repository,
            InMemoryDatasetSampleRepository(),
            VietnamPlateNormalizer(),
        ),
    )
    payload = {
        "text": "51H12346",
        "expectedRevision": 0,
        "note": "Verified against plate crop",
    }

    with TestClient(app) as client:
        reviewed = client.put(
            f"/api/events/{review_event.id}/plate-review",
            json=payload,
            headers={"X-Request-ID": "req-plate-review"},
        )
        retry = client.put(f"/api/events/{review_event.id}/plate-review", json=payload)
        conflict = client.put(
            f"/api/events/{review_event.id}/plate-review",
            json={"text": "51H12347", "expectedRevision": 0},
        )
        corrected_search = client.get(
            "/api/vehicles/search",
            params={"plate": "51H12346"},
        )
        prediction_search = client.get(
            "/api/vehicles/search",
            params={"plate": "51H12345"},
        )
        samples = client.get("/api/dataset-samples")
        invalid_dataset_cursor = client.get(
            "/api/dataset-samples",
            params={"cursor": "invalid-cursor"},
        )
        audits = client.get(
            "/api/audit-logs",
            params={
                "resourceType": "VEHICLE_EVENT",
                "resourceId": review_event.id,
            },
        )

    assert reviewed.status_code == 200
    assert reviewed.headers["cache-control"] == "no-store, private"
    body = reviewed.json()
    assert body["changed"] is True
    assert body["feedbackReason"] == "HUMAN_CORRECTION"
    assert body["datasetSampleId"].startswith("dss_")
    assert body["event"]["schemaVersion"] == 2
    assert body["event"]["status"] == "CONFIRMED"
    assert body["event"]["plate"]["prediction"]["normalized"] == "51H-123.45"
    assert body["event"]["plate"]["final"] == "51H-123.46"
    assert body["event"]["plate"]["review"]["revision"] == 1
    assert body["event"]["plate"]["review"]["reviewedBy"]["id"] == "development-admin"
    assert retry.status_code == 200
    assert retry.json()["changed"] is False
    assert conflict.status_code == 409
    assert corrected_search.json()["items"][0]["_id"] == review_event.id
    assert prediction_search.json()["items"] == []
    assert len(samples.json()["items"]) == 1
    assert samples.json()["items"][0]["label"] == "51H-123.46"
    assert samples.json()["items"][0]["imageKey"] == "vehicles/test/plate.jpg"
    assert invalid_dataset_cursor.status_code == 400
    assert [item["action"] for item in audits.json()["items"]] == ["PLATE_CORRECTED"]
    assert audits.json()["items"][0]["requestId"] == "req-plate-review"
