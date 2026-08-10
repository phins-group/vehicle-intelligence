"""Logical vehicle timeline and topology-aware journey generation."""

from __future__ import annotations

from datetime import datetime

from vehicle_intelligence.application.ports import (
    CameraTopologyRepository,
    VehicleEventRepository,
    VehicleIdentityRepository,
)
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import (
    JourneyObservation,
    JourneySegment,
    VehicleEvent,
    VehicleJourney,
)
from vehicle_intelligence.exceptions import IdentityNotFoundError


class VehicleJourneyService:
    def __init__(
        self,
        identities: VehicleIdentityRepository,
        events: VehicleEventRepository,
        topology: CameraTopologyRepository,
        config: IdentityConfig,
    ) -> None:
        self._identities = identities
        self._events = events
        self._topology = topology
        self._maximum_events = config.journey_event_limit

    async def timeline(
        self,
        vehicle_id: str,
        *,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[VehicleEvent, ...]:
        await self._required_identity(vehicle_id)
        requested = self._maximum_events if limit is None else limit
        if not 1 <= requested <= self._maximum_events:
            raise ValueError(f"journey limit must be in [1, {self._maximum_events}]")
        _validate_range(from_time, to_time)
        return await self._events.timeline(
            vehicle_id,
            from_time=from_time,
            to_time=to_time,
            limit=requested,
            ascending=True,
        )

    async def latest(self, vehicle_id: str) -> VehicleEvent | None:
        await self._required_identity(vehicle_id)
        events = await self._events.timeline(
            vehicle_id,
            limit=1,
            ascending=False,
        )
        return events[0] if events else None

    async def journey(
        self,
        vehicle_id: str,
        *,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        limit: int | None = None,
    ) -> VehicleJourney:
        requested = self._maximum_events if limit is None else limit
        if not 1 <= requested <= self._maximum_events:
            raise ValueError(f"journey limit must be in [1, {self._maximum_events}]")
        await self._required_identity(vehicle_id)
        _validate_range(from_time, to_time)
        values = await self._events.timeline(
            vehicle_id,
            from_time=from_time,
            to_time=to_time,
            limit=requested + 1,
            ascending=True,
        )
        truncated = len(values) > requested
        events = values[:requested]
        observations = tuple(_observation(event) for event in events)
        segments = tuple(
            [
                await self._segment(previous, current)
                for previous, current in zip(events, events[1:], strict=False)
            ]
        )
        return VehicleJourney(
            vehicle_id=vehicle_id,
            observations=observations,
            segments=segments,
            started_at=events[0].occurred_at if events else None,
            ended_at=events[-1].occurred_at if events else None,
            truncated=truncated,
        )

    async def _segment(
        self,
        previous: VehicleEvent,
        current: VehicleEvent,
    ) -> JourneySegment:
        elapsed = (current.occurred_at - previous.occurred_at).total_seconds()
        edges = await self._topology.list(
            from_camera_id=previous.camera.id,
            to_camera_id=current.camera.id,
            enabled_only=True,
            limit=1,
        )
        edge = edges[0] if edges else None
        return JourneySegment(
            from_event_id=previous.id,
            to_event_id=current.id,
            from_camera_id=previous.camera.id,
            to_camera_id=current.camera.id,
            departed_at=previous.occurred_at,
            arrived_at=current.occurred_at,
            elapsed_seconds=elapsed,
            topology_edge_id=edge.id if edge is not None else None,
            expected_minimum_seconds=(
                edge.minimum_travel_seconds if edge is not None else None
            ),
            expected_maximum_seconds=(
                edge.maximum_travel_seconds if edge is not None else None
            ),
            feasible=(
                edge.minimum_travel_seconds
                <= elapsed
                <= edge.maximum_travel_seconds
                if edge is not None
                else None
            ),
        )

    async def _required_identity(self, vehicle_id: str) -> None:
        if await self._identities.get(vehicle_id) is None:
            raise IdentityNotFoundError(f"vehicle identity not found: {vehicle_id}")


def _observation(event: VehicleEvent) -> JourneyObservation:
    return JourneyObservation(
        event_id=event.id,
        camera_id=event.camera.id,
        camera_name=event.camera.name,
        zone=event.camera.zone,
        occurred_at=event.occurred_at,
        event_type=event.event_type,
        direction=event.direction,
        status=event.status,
        plate=(event.plate.final_normalized if event.plate is not None else None),
        vehicle_type=event.vehicle.type,
    )


def _validate_range(from_time: datetime | None, to_time: datetime | None) -> None:
    for value in (from_time, to_time):
        if value is not None and value.tzinfo is None:
            raise ValueError("journey timestamps must include a timezone")
    if from_time is not None and to_time is not None and from_time > to_time:
        raise ValueError("journey time range is inverted")
