from pathlib import Path

from scripts.run_real_service_tests import (
    REAL_SERVICE_TEST_PATHS,
    REQUIRED_ENVIRONMENT,
    missing_environment,
)


def test_real_service_manifest_covers_every_environment_gated_integration_file() -> None:
    integration_directory = Path(__file__).resolve().parents[1] / "integration"
    gated_files = {
        path.relative_to(Path(__file__).resolve().parents[2]).as_posix()
        for path in integration_directory.glob("test_*.py")
        if any(name in path.read_text() for name in REQUIRED_ENVIRONMENT)
    }

    assert set(REAL_SERVICE_TEST_PATHS) == gated_files


def test_real_service_gate_requires_every_connection_setting() -> None:
    assert missing_environment({}) == REQUIRED_ENVIRONMENT
    assert (
        missing_environment(
            {
                "TEST_MONGODB_URI": "mongodb://127.0.0.1:27017",
                "TEST_REDIS_URL": "redis://127.0.0.1:6379/15",
                "TEST_MINIO_ENDPOINT": "127.0.0.1:9000",
            }
        )
        == ()
    )
