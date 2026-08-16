from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vehicle_intelligence.domain import (
    AITrace,
    CameraSnapshot,
    Direction,
    EventStatus,
    EventType,
    MediaReferences,
    ModelMetadata,
    PlateEvidence,
    VehicleEvent,
    VehicleEvidence,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    quality = parser.getgroup("quality gates")
    quality.addoption(
        "--fail-on-skip",
        action="store_true",
        default=False,
        help="Fail the test session when any selected test is skipped.",
    )


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    if not session.config.getoption("--fail-on-skip"):
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    skipped = reporter.stats.get("skipped", ()) if reporter is not None else ()
    if not skipped:
        return
    if reporter is not None:
        reporter.write_sep(
            "!",
            f"--fail-on-skip rejected {len(skipped)} skipped test(s)",
            red=True,
            bold=True,
        )
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.fixture
def model_metadata() -> ModelMetadata:
    return ModelMetadata(name="test-model", version="1")


@pytest.fixture
def sample_event(model_metadata: ModelMetadata) -> VehicleEvent:
    timestamp = datetime(2026, 8, 8, 13, 30, tzinfo=UTC)
    return VehicleEvent(
        id="evt_test",
        schema_version=1,
        camera=CameraSnapshot(id="gate-01", name="Main Gate", zone="ZONE_A"),
        track_id="gate-01:video-test:12",
        event_type=EventType.VEHICLE_ENTER,
        occurred_at=timestamp,
        created_at=timestamp,
        direction=Direction.ENTER,
        status=EventStatus.CONFIRMED,
        vehicle=VehicleEvidence(type="car", confidence=0.97),
        plate=PlateEvidence(
            raw="51H12345",
            normalized="51H-123.45",
            confidence=0.95,
            observation_count=4,
        ),
        media=MediaReferences(snapshot_key="vehicles/test/snapshot.jpg"),
        ai=AITrace(
            vehicle_detector=model_metadata,
            plate_detector=model_metadata,
            ocr=model_metadata,
        ),
    )
