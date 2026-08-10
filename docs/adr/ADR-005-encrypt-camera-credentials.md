# ADR-005: Encrypt camera credentials with camera-bound AEAD

**Status:** Accepted

RTSP URLs often contain usernames and passwords. The MongoDB camera adapter must
therefore store only authenticated ciphertext, while public API schemas, logs,
process arguments, and domain string representations must not reveal either the
plaintext or encrypted token.

The initial implementation uses AES-256-GCM behind `CredentialCipher`. Every
write generates a fresh 96-bit nonce. The camera ID is authenticated associated
data, preventing a valid token from being copied to a different camera record.
Tokens include a format version and key ID so migrations can distinguish future
formats, although this milestone supports one active key only.

The 32-byte key is supplied as URL-safe Base64 through environment/secret-manager
configuration and is never persisted in MongoDB. Mongo-backed camera management
fails closed when it is missing or invalid. The supervisor decrypts at the
process-composition boundary, passes the RTSP URL in a child-only environment
variable, and explicitly prevents the child from inheriting the master key.

Production key replacement requires a controlled re-encryption migration before
the old key is removed. Multi-key decryption and automated rotation are deferred
until an operational secret manager is selected.
