"""Regression checks for operational Docker Compose guarantees."""

from pathlib import Path

import yaml


def test_api_has_enough_time_for_graceful_lifespan_shutdown() -> None:
    compose_path = Path(__file__).parents[2] / "docker-compose.yml"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert document["services"]["api"]["stop_grace_period"] == "30s"


def test_container_healthcheck_uses_process_liveness() -> None:
    root = Path(__file__).parents[2]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")

    assert "http://127.0.0.1:8000/livez" in dockerfile


def test_edge_camera_has_enough_time_for_outbox_shutdown() -> None:
    compose_path = Path(__file__).parents[2] / "docker-compose.edge.yml"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    assert document["services"]["vision-edge"]["stop_grace_period"] == "75s"
