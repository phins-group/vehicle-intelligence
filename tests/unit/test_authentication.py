import hashlib

import pytest

from vehicle_intelligence.application.security import (
    Permission,
    RBACAuthorizer,
    StaticApiKeyAuthenticator,
)
from vehicle_intelligence.config import AuthConfig, AuthPrincipalConfig
from vehicle_intelligence.domain import UserRole


def digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def principal_config(identifier: str, role: str, token: str) -> AuthPrincipalConfig:
    return AuthPrincipalConfig(
        id=identifier,
        display_name=identifier.title(),
        role=role,
        key_sha256=digest(token),
    )


async def test_static_api_key_authenticator_returns_configured_principal() -> None:
    token = "admin-token-" + "a" * 40
    config = AuthConfig(
        enabled=True,
        principals=[principal_config("admin-01", "ADMIN", token)],
    )
    authenticator = StaticApiKeyAuthenticator(config)

    principal = await authenticator.authenticate(token)

    assert principal is not None
    assert principal.id == "admin-01"
    assert principal.role is UserRole.ADMIN
    assert await authenticator.authenticate("wrong-token-" + "b" * 40) is None
    assert await authenticator.authenticate("short") is None


def test_auth_config_fails_closed_without_active_admin_or_unique_verifiers() -> None:
    viewer_token = "viewer-token-" + "v" * 40
    with pytest.raises(ValueError, match="active ADMIN"):
        AuthConfig(
            enabled=True,
            principals=[principal_config("viewer-01", "VIEWER", viewer_token)],
        )

    shared_hash = digest("shared-token-" + "s" * 40)
    with pytest.raises(ValueError, match="key hashes must be unique"):
        AuthConfig(
            principals=[
                AuthPrincipalConfig(id="one", role="ADMIN", key_sha256=shared_hash),
                AuthPrincipalConfig(id="two", role="VIEWER", key_sha256=shared_hash),
            ]
        )


def test_invalid_auth_verifier_error_does_not_echo_supplied_value() -> None:
    accidental_raw_key = "accidental-raw-api-key-that-must-not-leak"
    with pytest.raises(ValueError) as captured:
        AuthPrincipalConfig(
            id="admin-01",
            role="ADMIN",
            key_sha256=accidental_raw_key,
        )
    assert accidental_raw_key not in str(captured.value)


async def test_rbac_permission_matrix_is_least_privilege() -> None:
    authorizer = RBACAuthorizer()
    tokens = {
        UserRole.ADMIN: "admin-" + "a" * 40,
        UserRole.OPERATOR: "operator-" + "o" * 40,
        UserRole.VIEWER: "viewer-" + "v" * 40,
    }
    configs = [
        principal_config(role.value.lower(), role.value, token)
        for role, token in tokens.items()
    ]
    auth = StaticApiKeyAuthenticator(AuthConfig(enabled=True, principals=configs))

    async def load(role: UserRole):
        principal = await auth.authenticate(tokens[role])
        assert principal is not None
        return principal

    admin = await load(UserRole.ADMIN)
    operator = await load(UserRole.OPERATOR)
    viewer = await load(UserRole.VIEWER)
    assert all(authorizer.allows(admin, permission) for permission in Permission)
    assert authorizer.allows(operator, Permission.MANAGE_ALERTS)
    assert authorizer.allows(operator, Permission.TEST_CAMERAS)
    assert authorizer.allows(operator, Permission.REVIEW_PLATES)
    assert authorizer.allows(operator, Permission.REVIEW_DATASETS)
    assert not authorizer.allows(operator, Permission.MANAGE_DATASETS)
    assert not authorizer.allows(operator, Permission.MANAGE_POLICIES)
    assert authorizer.allows(viewer, Permission.READ_PLATFORM)
    assert not authorizer.allows(viewer, Permission.MANAGE_ALERTS)
    assert not authorizer.allows(viewer, Permission.READ_AUDIT_LOGS)
    assert not authorizer.allows(viewer, Permission.REVIEW_PLATES)
    assert not authorizer.allows(viewer, Permission.REVIEW_DATASETS)
