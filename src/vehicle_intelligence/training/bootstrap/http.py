"""Bounded HTTPS client used only by pinned bootstrap source adapters."""

from __future__ import annotations

import json
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from vehicle_intelligence.exceptions import SampleDataAcquisitionError

_ALLOWED_HOST_SUFFIXES = (
    "huggingface.co",
    "hf.co",
    "googleapis.com",
    "amazonaws.com",
)


class BootstrapHttpClient(Protocol):
    def get_bytes(self, url: str, *, maximum_bytes: int) -> bytes: ...

    def get_json(self, url: str, *, maximum_bytes: int) -> dict[str, Any]: ...


class BoundedHttpClient:
    def __init__(self, *, timeout_seconds: float = 60.0) -> None:
        self._client = httpx.Client(
            follow_redirects=True,
            timeout=httpx.Timeout(timeout_seconds),
            headers={"User-Agent": "vehicle-intelligence-bootstrap/1"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BoundedHttpClient:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def get_bytes(self, url: str, *, maximum_bytes: int) -> bytes:
        _validate_url(url)
        if maximum_bytes < 1:
            raise SampleDataAcquisitionError("bootstrap response limit must be positive")
        try:
            with self._client.stream("GET", url) as response:
                response.raise_for_status()
                _validate_url(str(response.url))
                content_length = response.headers.get("content-length")
                if content_length is not None and int(content_length) > maximum_bytes:
                    raise SampleDataAcquisitionError("bootstrap response exceeds byte limit")
                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_bytes():
                    size += len(chunk)
                    if size > maximum_bytes:
                        raise SampleDataAcquisitionError("bootstrap response exceeds byte limit")
                    chunks.append(chunk)
        except SampleDataAcquisitionError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            raise SampleDataAcquisitionError("bootstrap HTTPS request failed") from exc
        return b"".join(chunks)

    def get_json(self, url: str, *, maximum_bytes: int) -> dict[str, Any]:
        raw = self.get_bytes(url, maximum_bytes=maximum_bytes)
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SampleDataAcquisitionError("bootstrap JSON response is invalid") from exc
        if not isinstance(value, dict):
            raise SampleDataAcquisitionError("bootstrap JSON root must be an object")
        return value


def _validate_url(value: str) -> None:
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or not any(
            host == suffix or host.endswith(f".{suffix}")
            for suffix in _ALLOWED_HOST_SUFFIXES
        )
    ):
        raise SampleDataAcquisitionError("bootstrap URL is outside the HTTPS allowlist")
