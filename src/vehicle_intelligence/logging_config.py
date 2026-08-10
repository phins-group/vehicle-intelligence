"""Small structured JSON logging setup."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

_STANDARD_FIELDS = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}
_SENSITIVE_FRAGMENTS = ("password", "secret", "access_key", "token", "rtsp")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_fields = _trace_fields()
        if trace_fields:
            payload.update(trace_fields)
        for key, value in record.__dict__.items():
            if key in _STANDARD_FIELDS or key.startswith("_"):
                continue
            payload[key] = self._redact(key, value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _redact(key: str, value: Any) -> Any:
        lowered = key.lower()
        if any(fragment in lowered for fragment in _SENSITIVE_FRAGMENTS):
            return "[REDACTED]"
        return value


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def _trace_fields() -> dict[str, str]:
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
    except (ImportError, RuntimeError):
        return {}
    if not context.is_valid:
        return {}
    return {
        "trace_id": format(context.trace_id, "032x"),
        "span_id": format(context.span_id, "016x"),
    }
