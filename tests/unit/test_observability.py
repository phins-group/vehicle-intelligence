import json
import logging
from datetime import UTC, datetime

from fastapi import FastAPI
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vehicle_intelligence.config import ObservabilityConfig
from vehicle_intelligence.domain import CameraHealth, CameraStatus
from vehicle_intelligence.infrastructure.observability.metrics import PrometheusMetrics
from vehicle_intelligence.infrastructure.observability.tracing import build_tracing_runtime
from vehicle_intelligence.logging_config import JsonFormatter


def camera_health() -> CameraHealth:
    now = datetime(2026, 8, 9, tzinfo=UTC)
    return CameraHealth(
        camera_id="gate-01",
        status=CameraStatus.ONLINE,
        source_fps=25,
        decode_fps=24,
        queue_size=1,
        dropped_frames=4,
        reconnect_count=2,
        connection_failures=1,
        stream_epoch=2,
        last_frame_at=now,
        updated_at=now,
        decoded_frames=100,
        sampled_frames=30,
        vehicle_detections=12,
        plate_detections=8,
        ocr_requests=6,
        ocr_success=5,
        events_created=3,
        track_count=2,
        inference_fps=6.2,
        vehicle_inference_latency_ms=14.5,
        plate_inference_latency_ms=7.5,
        ocr_latency_ms=20.5,
    )


def test_prometheus_metrics_use_normalized_bounded_labels_and_camera_counters() -> None:
    metrics = PrometheusMetrics(include_process_metrics=False)
    metrics.observe_http("get", "/api/events/{event_id}", 404, 0.012)
    payload = metrics.render([camera_health()]).decode()

    assert (
        'http_requests_total{method="GET",route="/api/events/{event_id}",status="404"} 1.0'
        in payload
    )
    assert 'camera_frames_total{camera_id="gate-01",kind="decoded"} 100.0' in payload
    assert 'camera_frames_dropped_total{camera_id="gate-01"} 4.0' in payload
    assert 'ocr_success_total{camera_id="gate-01"} 5.0' in payload
    assert 'inference_latency_ms{camera_id="gate-01",stage="vehicle"} 14.5' in payload
    assert "event-123" not in payload


def test_opentelemetry_fastapi_spans_and_json_logs_share_trace_context() -> None:
    exporter = InMemorySpanExporter()
    runtime = build_tracing_runtime(
        ObservabilityConfig(
            opentelemetry_enabled=True,
            otlp_traces_endpoint="http://collector.test/v1/traces",
            trace_sample_ratio=1,
        ),
        exporter=exporter,
    )
    assert runtime is not None
    app = FastAPI()

    @app.get("/items/{item_id}")
    async def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    runtime.instrument(app)
    with TestClient(app) as client:
        assert client.get("/items/bounded-id").status_code == 200

    tracer = runtime.provider.get_tracer("test")
    with tracer.start_as_current_span("log-context"):
        record = logging.LogRecord("test", logging.INFO, __file__, 1, "hello", (), None)
        rendered = json.loads(JsonFormatter().format(record))
    runtime.shutdown()

    spans = exporter.get_finished_spans()
    assert any(span.name == "GET /items/{item_id}" for span in spans)
    assert len(rendered["trace_id"]) == 32
    assert len(rendered["span_id"]) == 16
