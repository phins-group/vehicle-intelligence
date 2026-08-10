"""Bounded ONVIF discovery use case independent of UDP and FastAPI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

from vehicle_intelligence.application.ports import OnvifDiscoveryProvider
from vehicle_intelligence.domain import OnvifDiscoveredDevice


class OnvifDiscoveryService:
    def __init__(
        self,
        provider: OnvifDiscoveryProvider,
        maximum_results: int,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if maximum_results < 1:
            raise ValueError("ONVIF discovery result limit must be positive")
        self._provider = provider
        self._maximum_results = maximum_results
        self._clock = clock

    async def discover(self) -> tuple[OnvifDiscoveredDevice, ...]:
        discovered_at = self._clock()
        if discovered_at.tzinfo is None:
            raise ValueError("ONVIF discovery clock must be timezone-aware")
        devices = await self._provider.discover()
        unique: dict[str, OnvifDiscoveredDevice] = {}
        for device in devices:
            key = device.endpoint_reference.casefold()
            current = unique.get(key)
            if current is None or len(device.xaddrs) > len(current.xaddrs):
                unique[key] = replace(
                    device,
                    discovered_at=discovered_at.astimezone(UTC),
                )
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                (item.name or item.hardware or item.endpoint_reference).casefold(),
                item.endpoint_reference.casefold(),
            ),
        )
        return tuple(ordered[: self._maximum_results])
