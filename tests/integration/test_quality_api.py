import asyncio
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from vehicle_intelligence.application.model_quality import ModelQualityService
from vehicle_intelligence.config import ModelQualityConfig, load_settings
from vehicle_intelligence.infrastructure.persistence.memory import (
    InMemoryVehicleEventRepository,
)
from vehicle_intelligence.infrastructure.persistence.quality_memory import (
    InMemoryModelQualityRepository,
)
from vehicle_intelligence.infrastructure.persistence.review_memory import (
    InMemoryDatasetSampleRepository,
)
from vehicle_intelligence.interfaces.api import create_app


def test_model_quality_api_is_bounded_versioned_and_not_cached(sample_event) -> None:
    events = InMemoryVehicleEventRepository()
    samples = InMemoryDatasetSampleRepository()
    asyncio.run(events.save(sample_event))
    service = ModelQualityService(
        InMemoryModelQualityRepository(events, samples),
        ModelQualityConfig(default_window_days=30, maximum_window_days=365),
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )
    app = create_app(load_settings(), events, model_quality_service=service)

    with TestClient(app) as client:
        response = client.get(
            "/api/model-quality",
            params={"from": "2026-08-01T00:00:00Z", "to": "2026-08-10T00:00:00Z"},
        )
        invalid = client.get(
            "/api/model-quality",
            params={"from": "2025-01-01T00:00:00Z", "to": "2026-08-10T00:00:00Z"},
        )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, private"
    payload = response.json()
    assert payload["schemaVersion"] == 1
    assert payload["totals"]["eventCount"] == 1
    assert payload["models"][0]["model"] == {
        "name": "test-model",
        "version": "1",
        "hash": None,
    }
    assert payload["daily"][0]["day"] == "2026-08-08"
    assert invalid.status_code == 422
