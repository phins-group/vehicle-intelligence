from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MediaKind(StrEnum):
    SNAPSHOT = "snapshot"
    VEHICLE_CROP = "vehicle_crop"
    PLATE_CROP = "plate_crop"
    EVENT_CLIP = "event_clip"


@dataclass(frozen=True, slots=True)
class MediaRetentionClaim:
    event_id: str
    kind: MediaKind
    key: str
    lease_id: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not all((self.event_id.strip(), self.key.strip(), self.lease_id.strip())):
            raise ValueError("retention claim identifiers are required")
        if self.occurred_at.tzinfo is None:
            raise ValueError("retention claim timestamp must be timezone-aware")


@dataclass(frozen=True, slots=True)
class LifecycleReconcileResult:
    changed: bool
    managed_rules: int
    preserved_rules: int
