"""Bootstrap logical identities without equating a plate string to identity."""

from __future__ import annotations

import hashlib

from vehicle_intelligence.application.ports import (
    EventPolicyResult,
    VehicleEventIdentityLinker,
    VehicleEventPostProcessor,
    VehicleIdentityRepository,
)
from vehicle_intelligence.config import IdentityConfig
from vehicle_intelligence.domain import (
    PlateIdentitySignal,
    VehicleEvent,
    VehicleFingerprint,
    VehicleIdentity,
)
from vehicle_intelligence.exceptions import PersistenceError


def bootstrap_vehicle_id(event_id: str) -> str:
    return f"veh_{hashlib.sha256(event_id.encode()).hexdigest()[:32]}"


def fingerprint_id(event_id: str, schema_version: int) -> str:
    value = f"{event_id}|fingerprint|{schema_version}".encode()
    return f"vfp_{hashlib.sha256(value).hexdigest()[:32]}"


class VehicleIdentityProcessor:
    """Create one safe bootstrap identity per event; later ReID may merge them."""

    def __init__(
        self,
        repository: VehicleIdentityRepository,
        event_linker: VehicleEventIdentityLinker,
        config: IdentityConfig,
    ) -> None:
        self._repository = repository
        self._event_linker = event_linker
        self._config = config

    async def initialize(self) -> None:
        await self._repository.ensure_indexes()

    async def close(self) -> None:
        await self._repository.close()

    async def process(self, event: VehicleEvent) -> EventPolicyResult:
        vehicle_id = event.vehicle_id or bootstrap_vehicle_id(event.id)
        plate_text = event.plate.final_normalized if event.plate is not None else None
        plates = (
            (
                PlateIdentitySignal(
                    text=plate_text,
                    confidence=event.plate.confidence,
                    first_seen_at=event.occurred_at,
                    last_seen_at=event.occurred_at,
                ),
            )
            if plate_text is not None and event.plate is not None
            else ()
        )
        identity = VehicleIdentity(
            id=vehicle_id,
            primary_plate=plate_text,
            plates=plates,
            vehicle_type=event.vehicle.type,
            color=event.vehicle.color,
            first_seen_at=event.occurred_at,
            last_seen_at=event.occurred_at,
            observation_count=1,
        )
        fingerprint = VehicleFingerprint(
            id=fingerprint_id(event.id, self._config.fingerprint_schema_version),
            vehicle_id=vehicle_id,
            source_event_id=event.id,
            camera_id=event.camera.id,
            observed_at=event.occurred_at,
            vehicle_type=event.vehicle.type,
            vehicle_confidence=event.vehicle.confidence,
            plate=plate_text,
            plate_confidence=(event.plate.confidence if event.plate is not None else None),
            color=event.vehicle.color,
            schema_version=self._config.fingerprint_schema_version,
        )
        await self._repository.register_observation(identity, fingerprint)
        if not await self._event_linker.assign_vehicle_id(event.id, vehicle_id):
            raise PersistenceError(f"cannot link event to vehicle identity: {event.id}")
        return EventPolicyResult()


class CompositeVehicleEventPostProcessor:
    def __init__(self, processors: tuple[VehicleEventPostProcessor, ...]) -> None:
        if not processors:
            raise ValueError("composite post-processor requires at least one component")
        self._processors = processors

    async def initialize(self) -> None:
        initialized: list[VehicleEventPostProcessor] = []
        try:
            for processor in self._processors:
                await processor.initialize()
                initialized.append(processor)
        except Exception:
            for processor in reversed(initialized):
                await processor.close()
            raise

    async def process(self, event: VehicleEvent) -> EventPolicyResult:
        total = EventPolicyResult()
        for processor in self._processors:
            result = await processor.process(event)
            total = EventPolicyResult(
                matched_rules=total.matched_rules + result.matched_rules,
                actions_succeeded=total.actions_succeeded + result.actions_succeeded,
                actions_skipped=total.actions_skipped + result.actions_skipped,
            )
        return total

    async def close(self) -> None:
        first_error: Exception | None = None
        for processor in reversed(self._processors):
            try:
                await processor.close()
            except Exception as exc:
                first_error = first_error or exc
        if first_error is not None:
            raise first_error
