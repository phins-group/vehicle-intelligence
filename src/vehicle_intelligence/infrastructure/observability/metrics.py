"""Low-cardinality Prometheus metrics and latest camera telemetry rendering."""

from __future__ import annotations

from collections.abc import Iterable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

from vehicle_intelligence.domain import CameraHealth


class PrometheusMetrics:
    content_type = CONTENT_TYPE_LATEST

    def __init__(self, *, include_process_metrics: bool = True) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        if include_process_metrics:
            ProcessCollector(registry=self.registry)
            PlatformCollector(registry=self.registry)
            GCCollector(registry=self.registry)
        self.http_requests = Counter(
            "http_requests",
            "HTTP requests completed by the API.",
            ("method", "route", "status"),
            registry=self.registry,
        )
        self.http_duration = Histogram(
            "http_request_duration_seconds",
            "API request duration by normalized route.",
            ("method", "route"),
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )
        self.collection_errors = Counter(
            "observability_collection_errors",
            "Failures while collecting dynamic platform telemetry.",
            ("source",),
            registry=self.registry,
        )
        self.retention_objects_deleted = Counter(
            "retention_objects_deleted",
            "Media objects removed by retention.",
            ("kind",),
            registry=self.registry,
        )
        self.retention_object_failures = Counter(
            "retention_object_failures",
            "Media deletion failures during retention.",
            ("kind",),
            registry=self.registry,
        )
        self.retention_events_deleted = Counter(
            "retention_events_deleted",
            "Canonical events removed after coordinated media cleanup.",
            registry=self.registry,
        )
        self.retention_run_duration = Histogram(
            "retention_run_duration_seconds",
            "Duration of one bounded retention pass.",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 15, 30, 60),
            registry=self.registry,
        )

    def observe_http(
        self,
        method: str,
        route: str,
        status_code: int,
        duration_seconds: float,
    ) -> None:
        labels = (method.upper(), route, str(status_code))
        self.http_requests.labels(*labels).inc()
        self.http_duration.labels(method.upper(), route).observe(max(0.0, duration_seconds))

    def observe_retention(
        self,
        *,
        deleted_by_kind: dict[str, int],
        failed_by_kind: dict[str, int],
        events_deleted: int,
        duration_seconds: float,
    ) -> None:
        for kind, count in deleted_by_kind.items():
            self.retention_objects_deleted.labels(kind).inc(count)
        for kind, count in failed_by_kind.items():
            self.retention_object_failures.labels(kind).inc(count)
        self.retention_events_deleted.inc(events_deleted)
        self.retention_run_duration.observe(max(0.0, duration_seconds))

    def render(self, camera_health: Iterable[CameraHealth] = ()) -> bytes:
        return generate_latest(self.registry) + _render_camera_health(tuple(camera_health))


def _render_camera_health(items: tuple[CameraHealth, ...]) -> bytes:
    registry = CollectorRegistry(auto_describe=True)
    status = Gauge(
        "camera_online",
        "Whether the latest camera state is ONLINE.",
        ("camera_id",),
        registry=registry,
    )
    source_fps = Gauge(
        "camera_source_fps",
        "Latest reported source FPS.",
        ("camera_id",),
        registry=registry,
    )
    decode_fps = Gauge(
        "camera_decode_fps",
        "Latest reported decoder FPS.",
        ("camera_id",),
        registry=registry,
    )
    inference_fps = Gauge(
        "camera_inference_fps",
        "Latest sampled inference FPS.",
        ("camera_id",),
        registry=registry,
    )
    queue_size = Gauge(
        "camera_queue_size",
        "Latest bounded decode queue depth.",
        ("camera_id",),
        registry=registry,
    )
    track_count = Gauge(
        "track_count",
        "Current active tracks by camera.",
        ("camera_id",),
        registry=registry,
    )
    latency = Gauge(
        "inference_latency_ms",
        "Latest cumulative-average inference latency by stage.",
        ("camera_id", "stage"),
        registry=registry,
    )
    counters = {
        "camera_frames": Counter(
            "camera_frames",
            "Frames decoded or sampled by camera workers.",
            ("camera_id", "kind"),
            registry=registry,
        ),
        "camera_frames_dropped": Counter(
            "camera_frames_dropped",
            "Frames dropped by bounded camera queues.",
            ("camera_id",),
            registry=registry,
        ),
        "vehicle_detections": Counter(
            "vehicle_detections",
            "Vehicle detections produced by camera workers.",
            ("camera_id",),
            registry=registry,
        ),
        "plate_detections": Counter(
            "plate_detections",
            "Plate detections produced by camera workers.",
            ("camera_id",),
            registry=registry,
        ),
        "ocr_requests": Counter(
            "ocr_requests",
            "OCR requests attempted by camera workers.",
            ("camera_id",),
            registry=registry,
        ),
        "ocr_success": Counter(
            "ocr_success",
            "Valid OCR observations produced by camera workers.",
            ("camera_id",),
            registry=registry,
        ),
        "event_created": Counter(
            "event_created",
            "Final vehicle events created by camera workers.",
            ("camera_id",),
            registry=registry,
        ),
        "camera_reconnect": Counter(
            "camera_reconnect",
            "Successful camera reconnects.",
            ("camera_id",),
            registry=registry,
        ),
    }
    for health in items:
        camera_id = health.camera_id
        status.labels(camera_id).set(1 if health.status.value == "ONLINE" else 0)
        source_fps.labels(camera_id).set(health.source_fps)
        decode_fps.labels(camera_id).set(health.decode_fps)
        inference_fps.labels(camera_id).set(health.inference_fps)
        queue_size.labels(camera_id).set(health.queue_size)
        track_count.labels(camera_id).set(health.track_count)
        latency.labels(camera_id, "vehicle").set(health.vehicle_inference_latency_ms)
        latency.labels(camera_id, "plate").set(health.plate_inference_latency_ms)
        latency.labels(camera_id, "ocr").set(health.ocr_latency_ms)
        counters["camera_frames"].labels(camera_id, "decoded").inc(health.decoded_frames)
        counters["camera_frames"].labels(camera_id, "sampled").inc(health.sampled_frames)
        counters["camera_frames_dropped"].labels(camera_id).inc(health.dropped_frames)
        counters["vehicle_detections"].labels(camera_id).inc(health.vehicle_detections)
        counters["plate_detections"].labels(camera_id).inc(health.plate_detections)
        counters["ocr_requests"].labels(camera_id).inc(health.ocr_requests)
        counters["ocr_success"].labels(camera_id).inc(health.ocr_success)
        counters["event_created"].labels(camera_id).inc(health.events_created)
        counters["camera_reconnect"].labels(camera_id).inc(health.reconnect_count)
    return generate_latest(registry)
