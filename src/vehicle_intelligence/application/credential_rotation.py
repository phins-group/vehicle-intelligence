"""Online camera credential re-encryption without stream revision churn."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from vehicle_intelligence.application.ports import (
    CameraCredentialRotationRepository,
    RotatingCredentialCipher,
)


@dataclass(frozen=True, slots=True)
class CredentialRotationReport:
    scanned: int = 0
    rotated: int = 0
    already_current: int = 0
    conflicts: int = 0
    dry_run: bool = False
    active_key_id: str = ""


class CameraCredentialRotationService:
    def __init__(
        self,
        repository: CameraCredentialRotationRepository,
        cipher: RotatingCredentialCipher,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._repository = repository
        self._cipher = cipher
        self._clock = clock

    async def rotate(
        self,
        *,
        batch_size: int = 100,
        maximum_cameras: int = 10_000,
        dry_run: bool = False,
    ) -> CredentialRotationReport:
        if not 1 <= batch_size <= 1000 or not 1 <= maximum_cameras <= 1_000_000:
            raise ValueError("credential rotation bounds are invalid")
        scanned = rotated = current = conflicts = 0
        after: str | None = None
        while scanned < maximum_cameras:
            batch = await self._repository.list_encrypted_credentials(
                after,
                min(batch_size, maximum_cameras - scanned),
            )
            if not batch:
                break
            for credential in batch:
                after = credential.camera_id
                scanned += 1
                if not self._cipher.needs_rotation(credential.token):
                    current += 1
                    continue
                plaintext = self._cipher.decrypt(credential.token, credential.camera_id)
                if dry_run:
                    rotated += 1
                    continue
                replacement = self._cipher.encrypt(plaintext, credential.camera_id)
                plaintext = ""  # minimize lifetime of the immutable plaintext reference
                changed = await self._repository.replace_encrypted_credential(
                    credential.camera_id,
                    credential.token,
                    replacement,
                    self._now(),
                )
                if changed:
                    rotated += 1
                else:
                    conflicts += 1
            if len(batch) < batch_size:
                break
        return CredentialRotationReport(
            scanned=scanned,
            rotated=rotated,
            already_current=current,
            conflicts=conflicts,
            dry_run=dry_run,
            active_key_id=self._cipher.active_key_id,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise ValueError("credential rotation clock must be timezone-aware")
        return value.astimezone(UTC)
