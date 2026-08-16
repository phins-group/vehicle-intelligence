"""Optional OpenTelemetry SDK/OTLP composition kept outside the application layer."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from vehicle_intelligence.config import ObservabilityConfig


@dataclass(slots=True)
class TracingRuntime:
    provider: TracerProvider

    def instrument(self, app: FastAPI) -> None:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=self.provider,
            excluded_urls="/metrics,/api/system/health,/livez,/readyz",
        )

    def shutdown(self) -> None:
        self.provider.shutdown()


def build_tracing_runtime(
    config: ObservabilityConfig,
    *,
    exporter: SpanExporter | None = None,
) -> TracingRuntime | None:
    if not config.opentelemetry_enabled:
        return None
    resource = Resource.create(
        {
            SERVICE_NAME: config.service_name,
            SERVICE_VERSION: config.service_version,
        }
    )
    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(TraceIdRatioBased(config.trace_sample_ratio)),
    )
    if exporter is None:
        headers = _parse_headers(
            config.otlp_headers.get_secret_value() if config.otlp_headers else None
        )
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_traces_endpoint,
            headers=headers,
            timeout=config.export_timeout_seconds,
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    return TracingRuntime(provider)


def _parse_headers(value: str | None) -> dict[str, str] | None:
    if value is None or not value.strip():
        return None
    result: dict[str, str] = {}
    for item in value.split(","):
        key, separator, raw = item.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator or not key or not raw or any(char in key + raw for char in "\r\n"):
            raise ValueError("OTLP headers must be comma-separated key=value pairs")
        result[key] = raw
    return result
