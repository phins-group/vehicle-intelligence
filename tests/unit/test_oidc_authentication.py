from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from vehicle_intelligence.config import OIDCConfig
from vehicle_intelligence.domain import AuthenticationMethod, UserRole
from vehicle_intelligence.infrastructure.security.oidc import OIDCAuthenticator


class StaticSigningKeys:
    def __init__(self, key) -> None:
        self.key = key

    def get_signing_key_from_jwt(self, _token: str):
        return SimpleNamespace(key=self.key)


def config(**changes) -> OIDCConfig:
    values = {
        "issuer": "https://identity.example",
        "jwks_url": "https://identity.example/jwks",
        "audiences": ["vehicle-intelligence"],
        "role_mapping": {"platform-admin": "ADMIN", "platform-viewer": "VIEWER"},
        "leeway_seconds": 0,
    }
    values.update(changes)
    return OIDCConfig(**values)


def token(private_key, **claim_changes) -> str:
    now = datetime.now(UTC)
    claims = {
        "sub": "operator-01",
        "name": "Gate Operator",
        "iss": "https://identity.example",
        "aud": "vehicle-intelligence",
        "exp": now + timedelta(minutes=5),
        "roles": ["platform-viewer", "platform-admin"],
    }
    claims.update(claim_changes)
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-key"})


async def test_oidc_validates_signature_claims_and_maps_highest_role() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    authenticator = OIDCAuthenticator(
        config(),
        StaticSigningKeys(private_key.public_key()),
    )

    principal = await authenticator.authenticate(token(private_key))

    assert principal is not None
    assert principal.id == "operator-01"
    assert principal.role is UserRole.ADMIN
    assert principal.authentication_method is AuthenticationMethod.OIDC


async def test_oidc_fails_closed_for_wrong_issuer_audience_expiry_and_algorithm() -> None:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    authenticator = OIDCAuthenticator(
        config(),
        StaticSigningKeys(private_key.public_key()),
    )
    now = datetime.now(UTC)

    assert await authenticator.authenticate(token(private_key, iss="https://evil.example")) is None
    assert await authenticator.authenticate(token(private_key, aud="another-api")) is None
    expired = token(private_key, exp=now - timedelta(seconds=1))
    assert await authenticator.authenticate(expired) is None
    forged = jwt.encode(
        {
            "sub": "operator-01",
            "iss": "https://identity.example",
            "aud": "vehicle-intelligence",
            "exp": now + timedelta(minutes=5),
            "roles": ["platform-admin"],
        },
        "attacker-secret-that-is-at-least-32-bytes",
        algorithm="HS256",
    )
    assert await authenticator.authenticate(forged) is None
