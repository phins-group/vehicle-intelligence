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
_MAX_LOG_LINE_CHARACTERS = 8000
_MAX_LOG_SCAN_CHARACTERS = 32_000
_REDACTED = "[REDACTED]"
_CREDENTIAL_NAME = (
    r"(?:hf[_-]?token|hugging[ _-]?face[ _-]?hub[ _-]?token|access[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|api[ _-]?key|x[ _-]?api[ _-]?key|"
    r"password|passwd|pwd|client[_-]?secret|secret|token)"
)
_CREDENTIAL_VALUE = (
    rf"""(?:"[^"\r\n]{{0,{_MAX_LOG_SCAN_CHARACTERS}}}"|"""
    rf"""'[^'\r\n]{{0,{_MAX_LOG_SCAN_CHARACTERS}}}'|"""
    rf"""[^\s,;&"'<>]{{1,{_MAX_LOG_SCAN_CHARACTERS}}})"""
)
_CREDENTIAL_ASSIGNMENT = re.compile(
    rf"""(?ix)
    (?P<prefix>
        (?<![A-Za-z0-9_])
        ["']?{_CREDENTIAL_NAME}["']?
        (?![A-Za-z0-9_])
        \s*[:=]\s*
    )
    {_CREDENTIAL_VALUE}
    """
)
_CREDENTIAL_ARGUMENT = re.compile(
    rf"""(?ix)
    (?P<prefix>
        --{_CREDENTIAL_NAME}
        (?![A-Za-z0-9_])
        (?:\s+|=\s*)
    )
    {_CREDENTIAL_VALUE}
    """
)
_BEARER_CREDENTIAL = re.compile(
    rf"(?i)(\bbearer[ \t]+)"
    rf"(?=[^\s,;\"']{{8,}})[^\s,;\"']{{1,{_MAX_LOG_SCAN_CHARACTERS}}}"
)
_KNOWN_TOKEN = re.compile(
    r"(?i)\b(?:"
    r"hf_[A-Za-z0-9]{12,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"sk-[A-Za-z0-9_-]{16,}"
    r")\b"
)
_URL_WITH_QUERY = re.compile(rf"(?i)https?://[^\s<>'\"\[\](){{}}]{{1,{_MAX_LOG_SCAN_CHARACTERS}}}")
_SIGNED_QUERY_KEY = re.compile(
    r"(?i)(?:^|[&;])(?:"
    r"x-amz-(?:signature|credential|security-token)|"
    r"x-goog-signature|signature|sig|token|access[_-]?token|api[_-]?key|password"
    r")="
)


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
        self._token = (
            token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
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
            message=_safe_status_message(getattr(status, "message", None), self._token),
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
            return tuple(
                _safe_log_line(str(line), configured_secret=self._token) for line in values
            )[-tail:]
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
        if self._inspect_job is None or self._fetch_job_logs is None or self._cancel_job is None:
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
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _safe_status_message(value: object, configured_secret: str | None) -> str | None:
    if value is None:
        return None
    safe = _safe_log_line(str(value), configured_secret=configured_secret).strip()[:1000]
    return safe or None


def _safe_log_line(value: str, *, configured_secret: str | None = None) -> str:
    bounded = value[:_MAX_LOG_SCAN_CHARACTERS].rstrip("\r\n")
    without_ansi = _ANSI_ESCAPE.sub("", bounded)
    printable = "".join(
        character for character in without_ansi if character >= " " or character == "\t"
    )
    return _redact_sensitive_text(printable, configured_secret)[:_MAX_LOG_LINE_CHARACTERS]


def _redact_sensitive_text(value: str, configured_secret: str | None) -> str:
    redacted = _URL_WITH_QUERY.sub(_redact_signed_url, value)
    if configured_secret:
        redacted = redacted.replace(
            configured_secret[:_MAX_LOG_SCAN_CHARACTERS],
            _REDACTED,
        )
    redacted = _BEARER_CREDENTIAL.sub(lambda match: match.group(1) + _REDACTED, redacted)
    redacted = _CREDENTIAL_ASSIGNMENT.sub(_redact_credential_match, redacted)
    redacted = _CREDENTIAL_ARGUMENT.sub(_redact_credential_match, redacted)
    return _KNOWN_TOKEN.sub(_REDACTED, redacted)


def _redact_signed_url(match: re.Match[str]) -> str:
    value = match.group(0)
    parsed = urlsplit(value)
    if not _SIGNED_QUERY_KEY.search(parsed.query):
        return value
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, _REDACTED, parsed.fragment))


def _redact_credential_match(match: re.Match[str]) -> str:
    prefix = match.group("prefix")
    raw_value = match.group(0)[len(prefix) :]
    quote = raw_value[0] if raw_value[:1] in {'"', "'"} else ""
    return f"{prefix}{quote}{_REDACTED}{quote}"


def _datetime(value: object) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
