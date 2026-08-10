"""Append-only actor audit use cases with mandatory secret redaction."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from urllib.parse import urlsplit, urlunsplit

from vehicle_intelligence.application.ports import (
    AuditLogRepository,
    AuditPage,
    AuditQuery,
)
from vehicle_intelligence.domain import (
    AuditAction,
    AuditActor,
    AuditLog,
    AuditResourceType,
    Principal,
)
from vehicle_intelligence.exceptions import AuditNotFoundError, AuditWriteError, PersistenceError

_SENSITIVE_FRAGMENTS = (
    "authorization",
    "access_key",
    "accesskey",
    "api-key",
    "api_key",
    "apikey",
    "credential",
    "key_sha256",
    "password",
    "private_key",
    "privatekey",
    "rtsp",
    "secret",
    "token",
)
_MAX_DEPTH = 8
_MAX_ITEMS = 1000
_MAX_STRING_LENGTH = 4096


@dataclass(frozen=True, slots=True)
class AuditRecord:
    principal: Principal
    action: AuditAction
    resource_type: AuditResourceType
    resource_id: str
    request_id: str
    before: Mapping[str, object] | None = None
    after: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None


class AuditService:
    def __init__(
        self,
        repository: AuditLogRepository,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        id_factory: Callable[[], str] = lambda: f"aud_{uuid.uuid4().hex}",
    ) -> None:
        self._repository = repository
        self._clock = clock
        self._id_factory = id_factory

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def record(self, command: AuditRecord) -> AuditLog:
        now = self._clock()
        if now.tzinfo is None:
            raise ValueError("audit service clock must be timezone-aware")
        entry = AuditLog(
            id=self._id_factory(),
            actor=AuditActor(
                id=command.principal.id,
                display_name=command.principal.display_name,
                role=command.principal.role,
                authentication_method=command.principal.authentication_method,
            ),
            action=command.action,
            resource_type=command.resource_type,
            resource_id=command.resource_id,
            occurred_at=now.astimezone(UTC),
            request_id=command.request_id,
            before=_snapshot(command.before),
            after=_snapshot(command.after),
            metadata=_snapshot(command.metadata) or {},
        )
        try:
            await self._repository.append(entry)
        except PersistenceError as exc:
            raise AuditWriteError("required audit record could not be persisted") from exc
        return entry

    async def get(self, entry_id: str) -> AuditLog:
        entry = await self._repository.get(entry_id)
        if entry is None:
            raise AuditNotFoundError(f"audit record not found: {entry_id}")
        return entry

    async def list(self, query: AuditQuery) -> AuditPage:
        return await self._repository.list(query)


def _snapshot(value: Mapping[str, object] | None) -> dict[str, object] | None:
    if value is None:
        return None
    sanitized = _sanitize(value, 0)
    if not isinstance(sanitized, dict):
        raise ValueError("audit snapshot must be a mapping")
    return sanitized


def _sanitize(value: object, depth: int) -> object:
    if depth > _MAX_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _MAX_ITEMS:
                result["_truncated"] = True
                break
            name = str(key)
            if any(fragment in name.casefold() for fragment in _SENSITIVE_FRAGMENTS):
                result[name] = "[REDACTED]"
            else:
                result[name] = _sanitize(item, depth + 1)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = [_sanitize(item, depth + 1) for item in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            items.append("[TRUNCATED]")
        return items
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("audit snapshot datetimes must be timezone-aware")
        return value.astimezone(UTC)
    if isinstance(value, str):
        if value.casefold().startswith(("bearer ", "rtsp://", "rtsps://")):
            return "[REDACTED]"
        if value.casefold().startswith(("http://", "https://")):
            parsed = urlsplit(value)
            if parsed.username is not None or parsed.password is not None:
                return "[REDACTED]"
            return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))[
                :_MAX_STRING_LENGTH
            ]
        return value[:_MAX_STRING_LENGTH]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return "[UNSUPPORTED]"
