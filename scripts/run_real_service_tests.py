"""Run the required MongoDB, Redis, and MinIO integration-test gate."""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ENVIRONMENT = (
    "TEST_MONGODB_URI",
    "TEST_REDIS_URL",
    "TEST_MINIO_ENDPOINT",
)
REAL_SERVICE_TEST_PATHS = (
    "tests/integration/test_audit_mongo.py",
    "tests/integration/test_camera_mongo.py",
    "tests/integration/test_camera_topology_mongo.py",
    "tests/integration/test_identity_review_mongo.py",
    "tests/integration/test_live_monitor_redis.py",
    "tests/integration/test_minio_storage.py",
    "tests/integration/test_mongo_repository.py",
    "tests/integration/test_mongo_transactions.py",
    "tests/integration/test_policy_event_worker.py",
    "tests/integration/test_policy_mongo.py",
    "tests/integration/test_quality_mongo.py",
    "tests/integration/test_realtime_redis.py",
    "tests/integration/test_redis_streams.py",
    "tests/integration/test_retention_mongo_minio.py",
    "tests/integration/test_review_mongo.py",
    "tests/integration/test_vehicle_identity_mongo.py",
    "tests/integration/test_vehicle_journey_mongo.py",
)


def missing_environment(environment: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_ENVIRONMENT if not environment.get(name))


def main() -> int:
    missing = missing_environment(os.environ)
    if missing:
        names = ", ".join(missing)
        print(f"real-service test gate requires: {names}", file=sys.stderr)
        return 2

    command = [
        sys.executable,
        "-m",
        "pytest",
        "--fail-on-skip",
        "-ra",
        "-W",
        "error::starlette.exceptions.StarletteDeprecationWarning",
        "-W",
        "error::pytest.PytestUnhandledThreadExceptionWarning",
        *REAL_SERVICE_TEST_PATHS,
    ]
    return subprocess.run(command, cwd=REPOSITORY_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
