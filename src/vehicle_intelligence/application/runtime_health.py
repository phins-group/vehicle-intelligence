"""Bounded process-liveness and dependency-readiness assessment."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum

ReadinessProbe = Callable[[], Awaitable[bool]]


class RuntimeCheckStatus(StrEnum):
    READY = "ready"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    DISABLED = "disabled"


@dataclass(frozen=True, slots=True)
class RuntimeDependency:
    name: str
    required: bool
    probe: ReadinessProbe | None


@dataclass(frozen=True, slots=True)
class RuntimeCheck:
    status: RuntimeCheckStatus
    required: bool

    def to_document(self) -> dict[str, object]:
        return {"status": self.status.value, "required": self.required}


@dataclass(frozen=True, slots=True)
class RuntimeReadinessSnapshot:
    ready: bool
    checks: dict[str, RuntimeCheck]

    def to_document(self) -> dict[str, object]:
        return {
            "status": "ready" if self.ready else "not_ready",
            "checks": {name: check.to_document() for name, check in self.checks.items()},
        }


class RuntimeHealthService:
    """Cache and coalesce bounded dependency probes for a public readiness route."""

    def __init__(
        self,
        dependencies: tuple[RuntimeDependency, ...],
        *,
        probe_timeout_seconds: float = 2.0,
        cache_seconds: float = 1.0,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if probe_timeout_seconds <= 0:
            raise ValueError("readiness probe timeout must be positive")
        if cache_seconds < 0:
            raise ValueError("readiness cache duration cannot be negative")
        names = [dependency.name for dependency in dependencies]
        if any(not name for name in names) or len(names) != len(set(names)):
            raise ValueError("readiness dependency names must be non-empty and unique")
        self._dependencies = dependencies
        self._probe_timeout = probe_timeout_seconds
        self._cache_seconds = cache_seconds
        self._monotonic = monotonic_clock
        self._started = False
        self._lock = asyncio.Lock()
        self._probe_tasks: dict[str, asyncio.Task[bool]] = {}
        self._cached: RuntimeReadinessSnapshot | None = None
        self._cached_at = 0.0

    def start(self) -> None:
        self._started = True
        self._invalidate()

    async def stop(self) -> None:
        self._started = False
        self._invalidate()
        tasks = tuple(self._probe_tasks.values())
        self._probe_tasks.clear()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def assess(self) -> RuntimeReadinessSnapshot:
        async with self._lock:
            now = self._monotonic()
            if self._cached is not None and now - self._cached_at < self._cache_seconds:
                return self._cached
            snapshot = await self._assess_uncached()
            self._cached = snapshot
            self._cached_at = self._monotonic()
            return snapshot

    async def _assess_uncached(self) -> RuntimeReadinessSnapshot:
        application_status = (
            RuntimeCheckStatus.READY if self._started else RuntimeCheckStatus.UNAVAILABLE
        )
        checks = {"application": RuntimeCheck(application_status, required=True)}
        if not self._started:
            checks.update(
                {
                    dependency.name: self._inactive_check(dependency)
                    for dependency in self._dependencies
                }
            )
            return RuntimeReadinessSnapshot(ready=False, checks=checks)

        results = await asyncio.gather(
            *(self._assess_dependency(dependency) for dependency in self._dependencies)
        )
        checks.update(
            (dependency.name, check)
            for dependency, check in zip(self._dependencies, results, strict=True)
        )
        if not self._started:
            checks["application"] = RuntimeCheck(RuntimeCheckStatus.UNAVAILABLE, required=True)
        ready = all(
            check.status is RuntimeCheckStatus.READY for check in checks.values() if check.required
        )
        return RuntimeReadinessSnapshot(ready=ready, checks=checks)

    async def _assess_dependency(self, dependency: RuntimeDependency) -> RuntimeCheck:
        if dependency.probe is None:
            return RuntimeCheck(RuntimeCheckStatus.DISABLED, dependency.required)
        available = await self._run_probe(dependency)
        if available:
            status = RuntimeCheckStatus.READY
        elif dependency.required:
            status = RuntimeCheckStatus.UNAVAILABLE
        else:
            status = RuntimeCheckStatus.DEGRADED
        return RuntimeCheck(status, dependency.required)

    async def _run_probe(self, dependency: RuntimeDependency) -> bool:
        task = self._probe_tasks.get(dependency.name)
        if task is None:
            assert dependency.probe is not None
            task = asyncio.create_task(
                self._safe_probe(dependency.probe),
                name=f"readiness-{dependency.name}",
            )
            self._probe_tasks[dependency.name] = task
        try:
            return await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._probe_timeout,
            )
        except TimeoutError:
            return False
        finally:
            if task.done():
                self._probe_tasks.pop(dependency.name, None)

    @staticmethod
    async def _safe_probe(probe: ReadinessProbe) -> bool:
        try:
            return bool(await probe())
        except Exception:
            return False

    @staticmethod
    def _inactive_check(dependency: RuntimeDependency) -> RuntimeCheck:
        if dependency.probe is None:
            return RuntimeCheck(RuntimeCheckStatus.DISABLED, dependency.required)
        status = (
            RuntimeCheckStatus.UNAVAILABLE if dependency.required else RuntimeCheckStatus.DEGRADED
        )
        return RuntimeCheck(status, dependency.required)

    def _invalidate(self) -> None:
        self._cached = None
        self._cached_at = 0.0
