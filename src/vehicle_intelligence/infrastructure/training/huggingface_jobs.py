"""Hugging Face Jobs adapter behind the application training gateway."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vehicle_intelligence.domain.model_training import (
    RemoteTrainingJob,
    RemoteTrainingSubmission,
)
from vehicle_intelligence.exceptions import ModelRegistryError
from vehicle_intelligence.training.huggingface import HuggingFaceJobRunner

_ANSI_ESCAPE = re.compile(r"\x1b(?:[@-_]|\[[0-?]*[ -/]*[@-~])")


class HuggingFaceTrainingJobGateway:
    def __init__(
        self,
        *,
        token: str | None = None,
        runner: HuggingFaceJobRunner | None = None,
        inspect_job: Callable[..., Any] | None = None,
        fetch_job_logs: Callable[..., Iterable[str]] | None = None,
        cancel_job: Callable[..., None] | None = None,
    ) -> None:
        self._token = token or os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        self._runner = runner
        self._inspect_job = inspect_job
        self._fetch_job_logs = fetch_job_logs
        self._cancel_job = cancel_job

    def submit(self, request: RemoteTrainingSubmission) -> RemoteTrainingJob:
        runner = self._runner or HuggingFaceJobRunner()
        result = runner.submit(
            image=request.image,
            command=request.command,
            flavor=request.hardware_flavor,
            dataset_repo=request.dataset_repo_id,
            dataset_revision=request.dataset_revision,
            output_bucket=request.output_bucket,
            namespace=request.namespace,
            timeout_seconds=request.timeout_seconds,
            name=request.name,
            labels=request.labels,
        )
        if not result.job_id:
            raise ModelRegistryError("Hugging Face returned an empty training Job id")
        return RemoteTrainingJob(
            id=result.job_id,
            status=result.status or "SCHEDULING",
            url=_safe_job_url(result.url),
        )

    def inspect(self, job_id: str, namespace: str | None) -> RemoteTrainingJob:
        inspect_job, _, _ = self._controls()
        try:
            job = inspect_job(
                job_id=job_id,
                namespace=namespace,
                token=self._token,
            )
        except Exception as exc:
            raise ModelRegistryError("Hugging Face training Job inspection failed") from exc
        status = getattr(job, "status", None)
        return RemoteTrainingJob(
            id=str(getattr(job, "id", job_id)),
            status=_stage(getattr(status, "stage", None)),
            url=_safe_job_url(getattr(job, "url", None)),
            message=_optional_string(getattr(status, "message", None), 1000),
            started_at=_datetime(getattr(job, "started_at", None)),
            finished_at=_datetime(getattr(job, "finished_at", None)),
        )

    def logs(self, job_id: str, namespace: str | None, tail: int) -> tuple[str, ...]:
        _, fetch_job_logs, _ = self._controls()
        try:
            values = fetch_job_logs(
                job_id=job_id,
                namespace=namespace,
                follow=False,
                tail=tail,
                token=self._token,
            )
            return tuple(_safe_log_line(str(line)) for line in values)[-tail:]
        except Exception as exc:
            raise ModelRegistryError("Hugging Face training Job logs are unavailable") from exc

    def cancel(self, job_id: str, namespace: str | None) -> None:
        _, _, cancel_job = self._controls()
        try:
            cancel_job(job_id=job_id, namespace=namespace, token=self._token)
        except Exception as exc:
            raise ModelRegistryError("Hugging Face training Job cancellation failed") from exc

    def _controls(self) -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
        imported: tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]] | None = None
        if (
            self._inspect_job is None
            or self._fetch_job_logs is None
            or self._cancel_job is None
        ):
            imported = _control_dependencies()
        return (
            self._inspect_job or imported[0],
            self._fetch_job_logs or imported[1],
            self._cancel_job or imported[2],
        )


def _control_dependencies() -> tuple[Callable[..., Any], Callable[..., Any], Callable[..., Any]]:
    try:
        from huggingface_hub import cancel_job, fetch_job_logs, inspect_job
    except ImportError as exc:
        raise ModelRegistryError(
            "Hugging Face Jobs require `pip install -e '.[training]'`"
        ) from exc
    return inspect_job, fetch_job_logs, cancel_job


def _stage(value: object) -> str:
    raw = getattr(value, "value", value)
    text = str(raw or "SCHEDULING").strip().upper()
    return text if text else "SCHEDULING"


def _optional_string(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    safe = "".join(character for character in str(value) if character >= " " or character == "\t")
    return safe.strip()[:maximum] or None


def _safe_job_url(value: object) -> str | None:
    safe = _optional_string(value, 2048)
    if safe is None:
        return None
    parsed = urlsplit(safe)
    hostname = (parsed.hostname or "").lower()
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or (hostname != "huggingface.co" and not hostname.endswith(".huggingface.co"))
    ):
        return None
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def _safe_log_line(value: str) -> str:
    without_ansi = _ANSI_ESCAPE.sub("", value.rstrip("\r\n"))
    return "".join(
        character for character in without_ansi if character >= " " or character == "\t"
    )[:8000]


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
