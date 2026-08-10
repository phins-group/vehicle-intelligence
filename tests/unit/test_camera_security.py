import base64

import pytest

from vehicle_intelligence.config import SecurityConfig
from vehicle_intelligence.domain import SecretUri
from vehicle_intelligence.exceptions import ConfigurationError, CredentialEncryptionError
from vehicle_intelligence.infrastructure.security.aes_gcm import AesGcmCredentialCipher


def encoded_key(value: bytes = bytes(range(32))) -> str:
    return base64.urlsafe_b64encode(value).decode()


def test_aes_gcm_round_trip_is_randomized_and_camera_bound() -> None:
    cipher = AesGcmCredentialCipher.from_config(
        SecurityConfig(camera_credential_key=encoded_key(), camera_credential_key_id="key-1")
    )
    url = "rtsp://admin:top-secret@camera.example/live"

    first = cipher.encrypt(url, "gate-01")
    second = cipher.encrypt(url, "gate-01")

    assert first != second
    assert "top-secret" not in first
    assert cipher.decrypt(first, "gate-01") == url
    with pytest.raises(CredentialEncryptionError):
        cipher.decrypt(first, "gate-02")


def test_secret_uri_never_exposes_value_through_repr_or_str() -> None:
    value = SecretUri("rtsp://operator:hidden@camera.example/live")

    assert "hidden" not in repr(value)
    assert "hidden" not in str(value)
    assert value.reveal().endswith("/live")


def test_camera_encryption_key_must_be_32_bytes() -> None:
    with pytest.raises(ConfigurationError, match="32 bytes"):
        AesGcmCredentialCipher.from_config(
            SecurityConfig(camera_credential_key=encoded_key(b"too-short"))
        )


def test_keyring_decrypts_old_tokens_and_encrypts_only_with_active_key() -> None:
    old = AesGcmCredentialCipher.from_config(
        SecurityConfig(
            camera_credential_key=encoded_key(bytes(range(32))),
            camera_credential_key_id="old",
        )
    )
    token = old.encrypt("rtsp://camera/live", "gate-01")
    rotating = AesGcmCredentialCipher.from_config(
        SecurityConfig(
            camera_credential_keys=[
                {"id": "old", "key": encoded_key(bytes(range(32)))},
                {"id": "new", "key": encoded_key(bytes(reversed(range(32))))},
            ],
            camera_credential_active_key_id="new",
        )
    )

    assert rotating.decrypt(token, "gate-01") == "rtsp://camera/live"
    assert rotating.needs_rotation(token)
    replacement = rotating.encrypt("rtsp://camera/live", "gate-01")
    assert replacement.startswith("v1.new.")
    assert not rotating.needs_rotation(replacement)
