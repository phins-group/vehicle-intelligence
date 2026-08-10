from __future__ import annotations

import re
import uuid

from fastapi import Request

_REQUEST_ID = re.compile(r"[A-Za-z0-9._:-]{1,128}")


def resolve_request_id(request: Request) -> str:
    candidate = request.headers.get("X-Request-ID", "")
    if _REQUEST_ID.fullmatch(candidate):
        return candidate
    return f"req_{uuid.uuid4().hex}"


def request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if not isinstance(value, str) or not value:
        raise RuntimeError("request id middleware was not applied")
    return value

