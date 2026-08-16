"""Application boundary for durable vehicle-event finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from vehicle_intelligence.domain import VehicleEvent

MediaReferenceField = Literal[
    "snapshot_key",
    "vehicle_crop_key",
    "plate_crop_key",
]


@dataclass(frozen=True, slots=True)
class FinalizationMediaObject:
    """An encoded object that must be delivered with its vehicle event."""

    reference_field: MediaReferenceField
    key: str
    data: bytes
    content_type: str = "image/jpeg"

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("finalization media key cannot be empty")
        if not self.data:
            raise ValueError("finalization media data cannot be empty")
        if self.content_type != "image/jpeg":
            raise ValueError("finalization media must be encoded as image/jpeg")


@runtime_checkable
class FinalizationOutbox(Protocol):
    """Durably stage and asynchronously deliver a complete finalization unit."""

    async def initialize(self) -> None: ...

    async def stage(
        self,
        event: VehicleEvent,
        media: tuple[FinalizationMediaObject, ...],
    ) -> None: ...

    async def close(self) -> None: ...
