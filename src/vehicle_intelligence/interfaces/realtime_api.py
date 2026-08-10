"""Authorized SSE and WebSocket delivery of canonical vehicle events."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError
from starlette.websockets import WebSocketDisconnect

from vehicle_intelligence.application.ports import VehicleEventCodec
from vehicle_intelligence.application.realtime import (
    RealtimeDelivery,
    RealtimeEventService,
    RealtimeGap,
    RealtimeSubscriptionClosed,
)
from vehicle_intelligence.application.security import Permission
from vehicle_intelligence.config import RealtimeConfig
from vehicle_intelligence.domain import Principal
from vehicle_intelligence.interfaces.security import APISecurity

WEBSOCKET_UNAUTHORIZED = 4401
WEBSOCKET_FORBIDDEN = 4403
WEBSOCKET_BAD_REQUEST = 4400
WEBSOCKET_UNAVAILABLE = 1013


class WebSocketAuthenticate(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid", hide_input_in_errors=True)

    type: Literal["authenticate"]
    token: SecretStr
    last_event_id: str | None = Field(default=None, alias="lastEventId", max_length=128)


class RealtimeControlEnvelope(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str
    type: Literal[
        "system.realtime.ready",
        "system.realtime.heartbeat",
        "system.realtime.gap",
    ]
    schema_version: int = Field(default=1, alias="schemaVersion")
    occurred_at: datetime = Field(alias="occurredAt")
    source: Literal["api/realtime"] = "api/realtime"
    data: dict[str, object]


def build_realtime_router(
    service: RealtimeEventService | None,
    config: RealtimeConfig,
    security: APISecurity,
    codec: VehicleEventCodec,
) -> APIRouter:
    router = APIRouter(tags=["realtime"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/api/realtime/health")
    async def realtime_health(
        _principal: Principal = Depends(read_access),
    ) -> dict[str, object]:
        if service is None:
            return {"status": "DISABLED", "subscribers": 0}
        stats = service.stats
        return {
            "status": stats.source_state.value,
            "subscribers": stats.hub.subscribers,
            "eventsReceived": stats.hub.events_received,
            "eventsDistributed": stats.hub.events_distributed,
            "duplicateEvents": stats.hub.duplicate_events,
            "clientEventsDropped": stats.hub.client_events_dropped,
            "reconnectCount": stats.reconnect_count,
            "sourceFailures": stats.source_failures,
            "invalidMessages": stats.invalid_messages,
            "lastEventAt": stats.hub.last_event_at,
        }

    @router.get("/api/events/stream")
    async def stream_events(
        request: Request,
        _principal: Principal = Depends(read_access),
        last_event_id_header: Annotated[str | None, Header(alias="Last-Event-ID")] = None,
        last_event_id_query: Annotated[str | None, Query(alias="lastEventId")] = None,
    ) -> StreamingResponse:
        if service is None:
            raise HTTPException(status_code=503, detail="realtime delivery is disabled")
        last_event_id = _last_event_id(last_event_id_header, last_event_id_query)

        async def generate() -> AsyncIterator[str]:
            subscription = service.subscribe(last_event_id)
            try:
                yield "retry: 3000\n\n"
                while not await request.is_disconnected():
                    delivery = await subscription.receive(config.heartbeat_seconds)
                    if delivery is None:
                        yield f": heartbeat {datetime.now(UTC).isoformat()}\n\n"
                        continue
                    yield _sse_delivery(delivery, codec)
            except RealtimeSubscriptionClosed:
                return
            finally:
                service.unsubscribe(subscription)

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @router.websocket("/ws/events")
    async def websocket_events(websocket: WebSocket) -> None:
        await websocket.accept()
        authenticated = await _authenticate_websocket(websocket, security, config)
        if authenticated is None:
            return
        _principal, last_event_id = authenticated
        if service is None:
            await websocket.close(
                code=WEBSOCKET_UNAVAILABLE,
                reason="realtime delivery is disabled",
            )
            return

        subscription = service.subscribe(last_event_id)
        disconnect_task: asyncio.Task[None] | None = None
        try:
            await websocket.send_text(
                _control_json(
                    "system.realtime.ready",
                    {
                        "connectionId": subscription.id,
                        "delivery": "best-effort",
                        "recoveryEndpoint": "/api/events",
                    },
                )
            )
            disconnect_task = asyncio.create_task(_wait_for_disconnect(websocket))
            while True:
                delivery_task = asyncio.create_task(
                    subscription.receive(config.heartbeat_seconds)
                )
                done, _pending = await asyncio.wait(
                    {delivery_task, disconnect_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if disconnect_task in done:
                    delivery_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await delivery_task
                    return
                delivery = delivery_task.result()
                if delivery is None:
                    await websocket.send_text(
                        _control_json("system.realtime.heartbeat", {})
                    )
                    continue
                await websocket.send_text(_websocket_delivery(delivery, codec))
        except (RealtimeSubscriptionClosed, WebSocketDisconnect):
            return
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                with suppress(asyncio.CancelledError, RuntimeError, WebSocketDisconnect):
                    await disconnect_task
            service.unsubscribe(subscription)

    return router


async def _authenticate_websocket(
    websocket: WebSocket,
    security: APISecurity,
    config: RealtimeConfig,
) -> tuple[Principal, str | None] | None:
    authorization = websocket.headers.get("authorization")
    try:
        last_event_id = _last_event_id(
            websocket.headers.get("last-event-id"),
            websocket.query_params.get("lastEventId"),
        )
    except HTTPException:
        await websocket.close(code=WEBSOCKET_BAD_REQUEST, reason="invalid last event id")
        return None
    token: str
    if authorization is not None:
        token = _bearer_token(authorization) or ""
    elif security.enabled:
        try:
            raw = await _receive_json(websocket, config.websocket_auth_timeout_seconds)
            message = WebSocketAuthenticate.model_validate(raw)
        except (
            KeyError,
            RuntimeError,
            TimeoutError,
            TypeError,
            ValidationError,
            ValueError,
            WebSocketDisconnect,
        ):
            await websocket.close(code=WEBSOCKET_UNAUTHORIZED, reason="authentication required")
            return None
        token = message.token.get_secret_value()
        try:
            last_event_id = _last_event_id(last_event_id, message.last_event_id)
        except HTTPException:
            await websocket.close(code=WEBSOCKET_BAD_REQUEST, reason="invalid last event id")
            return None
    else:
        token = ""

    principal = await security.authenticate_token(token)
    if principal is None:
        await websocket.close(code=WEBSOCKET_UNAUTHORIZED, reason="authentication required")
        return None
    if not security.allows(principal, Permission.READ_PLATFORM):
        await websocket.close(code=WEBSOCKET_FORBIDDEN, reason="insufficient permission")
        return None
    return principal, last_event_id


async def _receive_json(websocket: WebSocket, timeout: float) -> object:
    text = await asyncio.wait_for(websocket.receive_text(), timeout=timeout)
    return json.loads(text)


async def _wait_for_disconnect(websocket: WebSocket) -> None:
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return


def _bearer_token(value: str) -> str | None:
    scheme, separator, token = value.partition(" ")
    if separator != " " or scheme.casefold() != "bearer" or not token:
        return None
    return token


def _last_event_id(primary: str | None, secondary: str | None) -> str | None:
    value = primary or secondary
    if value is None:
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > 128 or any(char.isspace() for char in stripped):
        raise HTTPException(status_code=422, detail="invalid last event id")
    return stripped


def _websocket_delivery(delivery: RealtimeDelivery, codec: VehicleEventCodec) -> str:
    if delivery.event is not None:
        return codec.encode(delivery.event)
    if delivery.gap is None:
        raise ValueError("invalid empty realtime delivery")
    return _gap_json(delivery.gap)


def _sse_delivery(delivery: RealtimeDelivery, codec: VehicleEventCodec) -> str:
    if delivery.event is not None:
        return f"id: {delivery.event.id}\ndata: {codec.encode(delivery.event)}\n\n"
    if delivery.gap is None:
        raise ValueError("invalid empty realtime delivery")
    return f"event: system.realtime.gap\ndata: {_gap_json(delivery.gap)}\n\n"


def _gap_json(gap: RealtimeGap) -> str:
    return _control_json(
        "system.realtime.gap",
        {
            "reason": gap.reason,
            "droppedEvents": gap.dropped_events,
            "lastAvailableEventId": gap.last_available_event_id,
            "recoveryEndpoint": "/api/events",
        },
    )


def _control_json(
    control_type: Literal[
        "system.realtime.ready",
        "system.realtime.heartbeat",
        "system.realtime.gap",
    ],
    data: dict[str, object],
) -> str:
    envelope = RealtimeControlEnvelope(
        id=f"ctl_{uuid4().hex}",
        type=control_type,
        occurredAt=datetime.now(UTC),
        data=data,
    )
    return envelope.model_dump_json(by_alias=True)
