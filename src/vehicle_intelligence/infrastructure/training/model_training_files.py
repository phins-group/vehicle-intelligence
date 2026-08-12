"""Atomic filesystem persistence for remote model-training run evidence."""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from vehicle_intelligence.config import ModelTrainingRuntimeConfig
from vehicle_intelligence.domain.model_training import (
    ModelTrainingParameters,
    ModelTrainingRun,
    ModelTrainingRunStatus,
)
from vehicle_intelligence.exceptions import (
    ModelTrainingConflictError,
    ModelTrainingNotFoundError,
    ModelTrainingStorageError,
)
from vehicle_intelligence.training.domain import DetectorRole

_RUN_ID = re.compile(r"^model-training-[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_ALLOWED_TRANSITIONS: dict[ModelTrainingRunStatus, frozenset[ModelTrainingRunStatus]] = {
    ModelTrainingRunStatus.QUEUED: frozenset(
        {
            ModelTrainingRunStatus.SUBMITTING,
            ModelTrainingRunStatus.FAILED,
            ModelTrainingRunStatus.CANCELED,
        }
    ),
    ModelTrainingRunStatus.SUBMITTING: frozenset(
        {
            ModelTrainingRunStatus.SCHEDULING,
            ModelTrainingRunStatus.RUNNING,
            ModelTrainingRunStatus.COMPLETED,
            ModelTrainingRunStatus.FAILED,
        }
    ),
    ModelTrainingRunStatus.SCHEDULING: frozenset(
        {
            ModelTrainingRunStatus.SCHEDULING,
            ModelTrainingRunStatus.RUNNING,
            ModelTrainingRunStatus.COMPLETED,
            ModelTrainingRunStatus.FAILED,
            ModelTrainingRunStatus.CANCELED,
        }
    ),
    ModelTrainingRunStatus.RUNNING: frozenset(
        {
            ModelTrainingRunStatus.RUNNING,
            ModelTrainingRunStatus.COMPLETED,
            ModelTrainingRunStatus.FAILED,
            ModelTrainingRunStatus.CANCELED,
        }
    ),
    ModelTrainingRunStatus.COMPLETED: frozenset(),
    ModelTrainingRunStatus.FAILED: frozenset(),
    ModelTrainingRunStatus.CANCELED: frozenset(),
}


class FileModelTrainingRunRepository:
    def __init__(self, config: ModelTrainingRuntimeConfig) -> None:
        self._config = config
        self._workspace = config.workspace_directory.expanduser().resolve()
        self._runs: dict[str, ModelTrainingRun] = {}
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        async with self._lock:
            self._runs = await asyncio.to_thread(self._load_runs)

    async def close(self) -> None:
        return None

    async def create(self, run: ModelTrainingRun) -> None:
        async with self._lock:
            if len(self._runs) >= self._config.maximum_runs:
                raise ModelTrainingConflictError("model training run history limit is reached")
            active = [item for item in self._runs.values() if item.status.active]
            if len(active) >= self._config.maximum_concurrent_runs:
                raise ModelTrainingConflictError(
                    "the configured concurrent model training limit is reached"
                )
            if any(
                item.model_name == run.model_name
                and item.model_version == run.model_version
                and item.status is ModelTrainingRunStatus.COMPLETED
                for item in self._runs.values()
            ):
                raise ModelTrainingConflictError(
                    "a completed model already uses this name and version"
                )
            if run.id in self._runs or run.status is not ModelTrainingRunStatus.QUEUED:
                raise ModelTrainingConflictError("model training run already exists or is invalid")
            await asyncio.to_thread(self._write, run, True)
            self._runs[run.id] = run

    async def save(self, run: ModelTrainingRun, expected_updated_at: datetime) -> None:
        async with self._lock:
            current = self._runs.get(run.id)
            if current is None:
                raise ModelTrainingNotFoundError(f"model training run not found: {run.id}")
            if current.updated_at != expected_updated_at:
                raise ModelTrainingConflictError("model training run was updated concurrently")
            if run.status not in _ALLOWED_TRANSITIONS[current.status]:
                raise ModelTrainingConflictError(
                    f"invalid model training transition: {current.status} -> {run.status}"
                )
            if _immutable_run_fields(run) != _immutable_run_fields(current):
                raise ModelTrainingConflictError("immutable model training evidence changed")
            if run.updated_at <= current.updated_at:
                raise ModelTrainingConflictError("model training update timestamp did not advance")
            await asyncio.to_thread(self._write, run, False)
            self._runs[run.id] = run

    async def get(self, run_id: str) -> ModelTrainingRun:
        async with self._lock:
            run = self._runs.get(run_id)
        if run is None:
            raise ModelTrainingNotFoundError(f"model training run not found: {run_id}")
        return run

    async def list(self) -> tuple[ModelTrainingRun, ...]:
        async with self._lock:
            values = tuple(self._runs.values())
        return tuple(sorted(values, key=lambda item: (item.created_at, item.id), reverse=True))

    def _load_runs(self) -> dict[str, ModelTrainingRun]:
        directory = self._runs_directory()
        if not directory.exists():
            return {}
        if not directory.is_dir() or directory.is_symlink():
            raise ModelTrainingStorageError("model training run directory is unsafe")
        paths = sorted(directory.glob("model-training-*.json"))
        if len(paths) > self._config.maximum_runs:
            raise ModelTrainingStorageError("model training run history exceeds configured limit")
        runs: dict[str, ModelTrainingRun] = {}
        for path in paths:
            if path.is_symlink() or not path.is_file():
                raise ModelTrainingStorageError("model training run path is unsafe")
            try:
                run = _run_from_json(json.loads(path.read_bytes()))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise ModelTrainingStorageError("model training run evidence is invalid") from exc
            if path.stem != run.id or run.id in runs:
                raise ModelTrainingStorageError("model training run filename is invalid")
            if run.status in {
                ModelTrainingRunStatus.QUEUED,
                ModelTrainingRunStatus.SUBMITTING,
            } and not run.remote_job_id:
                now = datetime.now(UTC)
                advanced = max(now, run.updated_at + timedelta(microseconds=1))
                run = replace(
                    run,
                    status=ModelTrainingRunStatus.FAILED,
                    updated_at=advanced,
                    finished_at=now,
                    error_code="INTERRUPTED_BEFORE_SUBMISSION",
                )
                self._write(run, False)
            runs[run.id] = run
        return runs

    def _runs_directory(self) -> Path:
        directory = (self._workspace / "runs").resolve()
        if not directory.is_relative_to(self._workspace):
            raise ModelTrainingStorageError("model training workspace path is unsafe")
        return directory

    def _write(self, run: ModelTrainingRun, create_only: bool) -> None:
        directory = self._runs_directory()
        directory.mkdir(parents=True, exist_ok=True)
        if not _RUN_ID.fullmatch(run.id):
            raise ModelTrainingStorageError("model training run id is unsafe")
        path = (directory / f"{run.id}.json").resolve()
        if not path.is_relative_to(directory) or (create_only and path.exists()):
            raise ModelTrainingConflictError("model training run file already exists or is unsafe")
        temporary = directory / f".{run.id}.{uuid.uuid4().hex}.tmp"
        payload = (
            json.dumps(_run_json(run), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise ModelTrainingStorageError("cannot persist model training run") from exc


def _run_json(run: ModelTrainingRun) -> dict[str, Any]:
    value = asdict(run)
    value["schema_version"] = 1
    value["role"] = run.role.value
    value["status"] = run.status.value
    for key in ("created_at", "updated_at", "started_at", "finished_at"):
        timestamp = value[key]
        value[key] = _timestamp(timestamp) if timestamp is not None else None
    return value


def _run_from_json(value: object) -> ModelTrainingRun:
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("model training run contract is invalid")
    run_id = str(value.get("id", ""))
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError("model training run id is invalid")
    hashes = (
        str(value.get("source_manifest_sha256", "")),
        str(value.get("export_manifest_sha256", "")),
    )
    if any(not _SHA256.fullmatch(item) for item in hashes):
        raise ValueError("model training evidence hash is invalid")
    commit_sha = str(value.get("dataset_commit_sha", ""))
    if not _COMMIT.fullmatch(commit_sha):
        raise ValueError("model training dataset commit is invalid")
    parameters = value.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("model training parameters are invalid")
    created_at = _datetime(value.get("created_at"))
    updated_at = _datetime(value.get("updated_at"))
    if updated_at < created_at:
        raise ValueError("model training timestamps are invalid")
    return ModelTrainingRun(
        id=run_id,
        role=DetectorRole(str(value["role"])),
        status=ModelTrainingRunStatus(str(value["status"])),
        source_id=_text(value, "source_id", 128),
        source_manifest_sha256=hashes[0],
        export_id=_text(value, "export_id", 128),
        export_manifest_sha256=hashes[1],
        dataset_repo_id=_text(value, "dataset_repo_id", 256),
        dataset_revision=_text(value, "dataset_revision", 128),
        dataset_commit_sha=commit_sha,
        model_repo_id=_text(value, "model_repo_id", 256),
        model_name=_text(value, "model_name", 128),
        model_version=_text(value, "model_version", 128),
        architecture=_text(value, "architecture", 128),
        parameters=ModelTrainingParameters(
            epochs=_integer(parameters, "epochs", 1, 10_000),
            batch_size=_integer(parameters, "batch_size", 1, 1024),
            workers=_integer(parameters, "workers", 0, 128),
            snapshot_epoch=_integer(parameters, "snapshot_epoch", 1, 10_000),
            timeout_seconds=_integer(parameters, "timeout_seconds", 60, 2_592_000),
            hardware_flavor=_nested_text(parameters, "hardware_flavor", 128),
        ),
        requested_by=_text(value, "requested_by", 128),
        dataset_rights_confirmed=value.get("dataset_rights_confirmed") is True,
        compute_cost_confirmed=value.get("compute_cost_confirmed") is True,
        restricted_data_confirmed=value.get("restricted_data_confirmed") is True,
        created_at=created_at,
        updated_at=updated_at,
        output_bucket=_text(value, "output_bucket", 256),
        output_path=_text(value, "output_path", 512),
        remote_job_id=_optional_text(value.get("remote_job_id"), 256),
        remote_job_url=_optional_text(value.get("remote_job_url"), 2048),
        remote_message=_optional_text(value.get("remote_message"), 1000),
        started_at=_optional_datetime(value.get("started_at")),
        finished_at=_optional_datetime(value.get("finished_at")),
        error_code=_optional_text(value.get("error_code"), 64),
    )


def _immutable_run_fields(run: ModelTrainingRun) -> tuple[object, ...]:
    return (
        run.id,
        run.role,
        run.source_id,
        run.source_manifest_sha256,
        run.export_id,
        run.export_manifest_sha256,
        run.dataset_repo_id,
        run.dataset_revision,
        run.dataset_commit_sha,
        run.model_repo_id,
        run.model_name,
        run.model_version,
        run.architecture,
        run.parameters,
        run.requested_by,
        run.dataset_rights_confirmed,
        run.compute_cost_confirmed,
        run.restricted_data_confirmed,
        run.created_at,
        run.output_bucket,
        run.output_path,
    )


def _text(value: dict[str, Any], key: str, maximum: int) -> str:
    return _nested_text(value, key, maximum)


def _nested_text(value: dict[str, Any], key: str, maximum: int) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip() or len(item) > maximum or "\x00" in item:
        raise ValueError(f"model training field is invalid: {key}")
    return item


def _optional_text(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or "\x00" in value:
        raise ValueError("optional model training field is invalid")
    return value


def _integer(value: dict[str, Any], key: str, minimum: int, maximum: int) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int) or not minimum <= item <= maximum:
        raise ValueError(f"model training integer is invalid: {key}")
    return item


def _datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("model training timestamp is invalid")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("model training timestamp must be timezone-aware")
    return result.astimezone(UTC)


def _optional_datetime(value: object) -> datetime | None:
    return None if value is None else _datetime(value)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        raise ModelTrainingStorageError("model training timestamp must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
