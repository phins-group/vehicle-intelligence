"""Validated model-quality reporting use case."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from vehicle_intelligence.application.ports import ModelQualityRepository
from vehicle_intelligence.config import ModelQualityConfig
from vehicle_intelligence.domain import ModelQualityReport


class ModelQualityService:
    def __init__(
        self,
        repository: ModelQualityRepository,
        config: ModelQualityConfig,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._config = config
        self._clock = clock

    async def report(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> ModelQualityReport:
        generated_at = self._clock()
        if generated_at.tzinfo is None:
            raise ValueError("quality service clock must be timezone-aware")
        generated_at = generated_at.astimezone(UTC)
        if any(value is not None and value.tzinfo is None for value in (from_time, to_time)):
            raise ValueError("quality report timestamps must include a timezone")
        end = (to_time or generated_at).astimezone(UTC)
        start = (
            from_time.astimezone(UTC)
            if from_time is not None
            else end - timedelta(days=self._config.default_window_days)
        )
        if end <= start:
            raise ValueError("quality report 'to' must be later than 'from'")
        if end - start > timedelta(days=self._config.maximum_window_days):
            raise ValueError("quality report window exceeds configured maximum")
        return await self._repository.summarize(
            start,
            end,
            generated_at,
            self._config.maximum_models,
        )

    async def close(self) -> None:
        await self._repository.close()
