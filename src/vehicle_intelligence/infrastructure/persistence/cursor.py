from __future__ import annotations

import base64
import json
from datetime import datetime

from vehicle_intelligence.exceptions import InvalidCursorError


def encode_cursor(occurred_at: datetime, event_id: str) -> str:
    payload = json.dumps(
        {"occurredAt": occurred_at.isoformat(), "id": event_id},
        separators=(",", ":"),
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


def decode_cursor(value: str) -> tuple[datetime, str]:
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode())
        occurred_at = datetime.fromisoformat(payload["occurredAt"])
        event_id = str(payload["id"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise InvalidCursorError("invalid cursor") from exc
    if occurred_at.tzinfo is None:
        raise InvalidCursorError("cursor timestamp must be timezone-aware")
    return occurred_at, event_id
