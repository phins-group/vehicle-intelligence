from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from vehicle_intelligence.domain.enums import VehicleIdentityStatus


@dataclass(frozen=True, slots=True)
class EmbeddingModel:
    name: str
    version: str
    dimension: int
    model_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or self.dimension < 1:
            raise ValueError("embedding model identity and dimension are required")
        if self.dimension > 65_536:
            raise ValueError("embedding dimension is unreasonably large")

    @property
    def key(self) -> str:
        return f"{self.name}:{self.version}"


@dataclass(frozen=True, slots=True)
class EmbeddingReference:
    id: str
    model: EmbeddingModel

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("embedding reference id is required")


@dataclass(frozen=True, slots=True)
class EmbeddingVector:
    id: str
    model: EmbeddingModel
    values: tuple[float, ...]
    created_at: datetime
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id.strip() or self.created_at.tzinfo is None:
            raise ValueError("embedding vector identity and aware timestamp are required")
        if len(self.values) != self.model.dimension:
            raise ValueError("embedding vector dimension does not match model metadata")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("embedding vector values must be finite")
        norm = math.sqrt(sum(value * value for value in self.values))
        if norm <= 0:
            raise ValueError("embedding vector cannot have zero norm")

    @property
    def normalized_values(self) -> tuple[float, ...]:
        norm = math.sqrt(sum(value * value for value in self.values))
        return tuple(value / norm for value in self.values)


@dataclass(frozen=True, slots=True)
class VectorNeighbor:
    vector_id: str
    score: float

    def __post_init__(self) -> None:
        if not self.vector_id.strip() or not -1 <= self.score <= 1:
            raise ValueError("vector neighbor is invalid")


@dataclass(frozen=True, slots=True)
class PlateIdentitySignal:
    text: str
    confidence: float
    first_seen_at: datetime
    last_seen_at: datetime

    def __post_init__(self) -> None:
        if not self.text.strip() or not 0 <= self.confidence <= 1:
            raise ValueError("plate identity signal is invalid")
        if self.first_seen_at.tzinfo is None or self.last_seen_at.tzinfo is None:
            raise ValueError("plate identity timestamps must be timezone-aware")
        if self.last_seen_at < self.first_seen_at:
            raise ValueError("plate identity time range is inverted")


@dataclass(frozen=True, slots=True)
class VehicleFingerprint:
    id: str
    vehicle_id: str
    source_event_id: str
    camera_id: str
    observed_at: datetime
    vehicle_type: str
    vehicle_confidence: float
    plate: str | None = None
    plate_confidence: float | None = None
    color: str | None = None
    embedding: EmbeddingReference | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if any(
            not value.strip()
            for value in (self.id, self.vehicle_id, self.source_event_id, self.camera_id)
        ):
            raise ValueError("vehicle fingerprint identifiers are required")
        if self.observed_at.tzinfo is None or not 0 <= self.vehicle_confidence <= 1:
            raise ValueError("vehicle fingerprint observation is invalid")
        if self.plate_confidence is not None and not 0 <= self.plate_confidence <= 1:
            raise ValueError("fingerprint plate confidence is invalid")
        if (self.plate is None) != (self.plate_confidence is None):
            raise ValueError("fingerprint plate and confidence must be present together")
        if self.schema_version < 1:
            raise ValueError("fingerprint schema version must be positive")


@dataclass(frozen=True, slots=True)
class VehicleIdentity:
    id: str
    primary_plate: str | None
    plates: tuple[PlateIdentitySignal, ...]
    vehicle_type: str | None
    color: str | None
    first_seen_at: datetime
    last_seen_at: datetime
    observation_count: int
    status: VehicleIdentityStatus = VehicleIdentityStatus.ACTIVE
    revision: int = 1
    schema_version: int = 1
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            not self.id.strip()
            or self.first_seen_at.tzinfo is None
            or self.last_seen_at.tzinfo is None
        ):
            raise ValueError("vehicle identity and aware timestamps are required")
        if self.last_seen_at < self.first_seen_at or self.observation_count < 1:
            raise ValueError("vehicle identity lifecycle is invalid")
        if self.revision < 1 or self.schema_version < 1:
            raise ValueError("vehicle identity versions must be positive")
        if len(self.plates) > 16:
            raise ValueError("vehicle identity plate aliases are bounded to 16")
        if self.primary_plate is not None and self.primary_plate not in {
            plate.text for plate in self.plates
        }:
            raise ValueError("primary plate must be represented by a plate signal")
