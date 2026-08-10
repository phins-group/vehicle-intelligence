"""AES-256-GCM credential protection with camera-bound associated data."""

from __future__ import annotations

import base64
import secrets

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from vehicle_intelligence.config import SecurityConfig
from vehicle_intelligence.exceptions import ConfigurationError, CredentialEncryptionError

TOKEN_VERSION = "v1"
NONCE_BYTES = 12


def _encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _decode(value: str) -> bytes:
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


class AesGcmCredentialCipher:
    """Encrypt with one active key and decrypt with every configured retained key."""

    def __init__(
        self,
        key: bytes,
        key_id: str,
        *,
        decryption_keys: dict[str, bytes] | None = None,
    ) -> None:
        keys = dict(decryption_keys or {})
        keys[key_id] = key
        if not key_id or "." in key_id:
            raise ConfigurationError("camera credential key id is invalid")
        if any(len(value) != 32 for value in keys.values()):
            raise ConfigurationError("camera credential key must decode to exactly 32 bytes")
        if any(not item or "." in item for item in keys):
            raise ConfigurationError("camera credential key id is invalid")
        self._keys = {item: AESGCM(value) for item, value in keys.items()}
        self._key_id = key_id

    @classmethod
    def from_config(cls, config: SecurityConfig) -> AesGcmCredentialCipher:
        encoded_keys = {
            item.id: item.key.get_secret_value() for item in config.camera_credential_keys
        }
        if config.camera_credential_key is not None:
            encoded_keys.setdefault(
                config.camera_credential_key_id,
                config.camera_credential_key.get_secret_value(),
            )
        if not encoded_keys:
            raise ConfigurationError("camera credential encryption key is not configured")
        active_id = config.camera_credential_active_key_id or config.camera_credential_key_id
        if active_id not in encoded_keys:
            raise ConfigurationError("active camera credential key is not configured")
        try:
            keys = {key_id: _decode(value) for key_id, value in encoded_keys.items()}
        except (ValueError, TypeError) as exc:
            raise ConfigurationError("camera credential key must be URL-safe base64") from exc
        return cls(keys[active_id], active_id, decryption_keys=keys)

    def encrypt(self, plaintext: str, context: str) -> str:
        if not plaintext or not context:
            raise CredentialEncryptionError("credential plaintext and context are required")
        nonce = secrets.token_bytes(NONCE_BYTES)
        ciphertext = self._keys[self._key_id].encrypt(
            nonce, plaintext.encode("utf-8"), self._aad(context)
        )
        return ".".join((TOKEN_VERSION, self._key_id, _encode(nonce), _encode(ciphertext)))

    def decrypt(self, token: str, context: str) -> str:
        try:
            version, key_id, nonce_value, ciphertext_value = token.split(".", maxsplit=3)
            if version != TOKEN_VERSION or key_id not in self._keys:
                raise ValueError("unsupported credential token")
            nonce = _decode(nonce_value)
            ciphertext = _decode(ciphertext_value)
            if len(nonce) != NONCE_BYTES:
                raise ValueError("invalid credential nonce")
            plaintext = self._keys[key_id].decrypt(nonce, ciphertext, self._aad(context))
            return plaintext.decode("utf-8")
        except (InvalidTag, UnicodeDecodeError, ValueError) as exc:
            raise CredentialEncryptionError("camera credential cannot be decrypted") from exc

    @property
    def active_key_id(self) -> str:
        return self._key_id

    def token_key_id(self, token: str) -> str:
        try:
            version, key_id, _nonce, _ciphertext = token.split(".", maxsplit=3)
        except ValueError as exc:
            raise CredentialEncryptionError("camera credential token is malformed") from exc
        if version != TOKEN_VERSION or not key_id:
            raise CredentialEncryptionError("camera credential token is unsupported")
        return key_id

    def needs_rotation(self, token: str) -> bool:
        return self.token_key_id(token) != self._key_id

    @staticmethod
    def _aad(context: str) -> bytes:
        return f"vehicle-intelligence:camera-rtsp:{context}".encode()
