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


def test_production_compose_uses_immutable_images_file_secrets_and_hardening() -> None:
    compose_path = Path(__file__).parents[2] / "docker-compose.production.yml"
    document = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = document["services"]
    assert "build" not in services["api"]
    assert "build" not in services["web"]
    assert services["api"]["image"].endswith(
        "@sha256:${VIP_API_IMAGE_SHA256:?Set the API image digest}"
    )
    assert services["web"]["image"].endswith(
        "@sha256:${VIP_WEB_IMAGE_SHA256:?Set the web image digest}"
    )
    assert services["api"]["read_only"] is True
    assert services["web"]["read_only"] is True
    assert services["api"]["cap_drop"] == ["ALL"]
    assert services["api"]["environment"]["VIP_AUTH__PROVIDER"] == "oidc"
    assert services["api"]["environment"]["VIP_MONGODB__URI_FILE"] == ("/run/secrets/mongodb_uri")
    assert set(services["api"]["secrets"]) == {
        "mongodb_uri",
        "redis_url",
        "minio_access_key",
        "minio_secret_key",
        "camera_credential_key",
    }


def test_web_gateway_has_rate_and_connection_limits() -> None:
    root = Path(__file__).parents[2]
    nginx = (root / "apps" / "web" / "nginx.conf").read_text(encoding="utf-8")

    assert "limit_req_zone" in nginx
    assert "limit_req zone=api_per_client" in nginx
    assert "limit_conn connections_per_client" in nginx
    assert "limit_req_status 429" in nginx
    assert "proxy_ssl_server_name on" in nginx
    assert "proxy_ssl_verify on" in nginx
    assert "proxy_pass ${WEB_API_UPSTREAM}" in nginx
    assert "proxy_pass ${WEB_MEDIA_UPSTREAM}" in nginx
    assert "${WEB_REALTIME_CONNECT_SRC}" in nginx


def test_production_collector_uses_fsynced_byte_bounded_persistent_queue() -> None:
    root = Path(__file__).parents[2]
    document = yaml.safe_load(
        (root / "infrastructure" / "otel" / "collector.production.yml").read_text(encoding="utf-8")
    )

    storage = document["extensions"]["file_storage"]
    queue = document["exporters"]["otlphttp/traces"]["sending_queue"]
    readers = document["service"]["telemetry"]["metrics"]["readers"]
    assert storage["create_directory"] is True
    assert storage["fsync"] is True
    assert queue["storage"] == "file_storage"
    assert queue["sizer"] == "bytes"
    assert queue["queue_size"] == 256 * 1024 * 1024
    assert readers[0]["pull"]["exporter"]["prometheus"]["port"] == 8888
