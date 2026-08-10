import base64
from datetime import UTC, datetime

from vehicle_intelligence.application.credential_rotation import (
    CameraCredentialRotationService,
)
from vehicle_intelligence.application.ports import EncryptedCameraCredential
from vehicle_intelligence.config import SecurityConfig
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher


def key(value: int) -> str:
    return base64.urlsafe_b64encode(bytes([value]) * 32).decode()


class CredentialRepository:
    def __init__(self, values: dict[str, str]) -> None:
        self.values = values
        self.force_conflict: set[str] = set()

    async def list_encrypted_credentials(self, after_camera_id, limit):
        ids = sorted(
            item for item in self.values if after_camera_id is None or item > after_camera_id
        )
        return tuple(
            EncryptedCameraCredential(item, self.values[item]) for item in ids[:limit]
        )

    async def replace_encrypted_credential(
        self,
        camera_id,
        expected_token,
        replacement_token,
        rotated_at,
    ):
        assert rotated_at.tzinfo is not None
        if camera_id in self.force_conflict or self.values[camera_id] != expected_token:
            return False
        self.values[camera_id] = replacement_token
        return True

    async def close(self):
        return None


async def test_rotation_reencrypts_old_key_with_cas_and_keeps_current_token() -> None:
    old = AesGcmCredentialCipher.from_config(
        SecurityConfig(camera_credential_key=key(1), camera_credential_key_id="old")
    )
    rotating = AesGcmCredentialCipher.from_config(
        SecurityConfig(
            camera_credential_keys=[
                {"id": "old", "key": key(1)},
                {"id": "new", "key": key(2)},
            ],
            camera_credential_active_key_id="new",
        )
    )
    current = rotating.encrypt("rtsp://two", "camera-02")
    repository = CredentialRepository(
        {
            "camera-01": old.encrypt("rtsp://one", "camera-01"),
            "camera-02": current,
        }
    )
    service = CameraCredentialRotationService(
        repository,
        rotating,
        clock=lambda: datetime(2026, 8, 10, tzinfo=UTC),
    )

    report = await service.rotate(batch_size=1)

    assert report.scanned == 2
    assert report.rotated == 1
    assert report.already_current == 1
    assert rotating.decrypt(repository.values["camera-01"], "camera-01") == "rtsp://one"
    assert repository.values["camera-02"] == current
