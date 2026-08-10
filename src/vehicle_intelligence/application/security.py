"""Authentication and permission policy independent of FastAPI."""

from __future__ import annotations

import hashlib
import hmac
from enum import StrEnum

from vehicle_intelligence.config import AuthConfig
from vehicle_intelligence.domain import AuthenticationMethod, Principal, UserRole


class Permission(StrEnum):
    READ_PLATFORM = "READ_PLATFORM"
    MANAGE_CAMERAS = "MANAGE_CAMERAS"
    TEST_CAMERAS = "TEST_CAMERAS"
    MANAGE_POLICIES = "MANAGE_POLICIES"
    MANAGE_ALERTS = "MANAGE_ALERTS"
    REVIEW_PLATES = "REVIEW_PLATES"
    READ_AUDIT_LOGS = "READ_AUDIT_LOGS"
    MANAGE_TOPOLOGY = "MANAGE_TOPOLOGY"
    REVIEW_IDENTITIES = "REVIEW_IDENTITIES"


_ROLE_PERMISSIONS: dict[UserRole, frozenset[Permission]] = {
    UserRole.VIEWER: frozenset({Permission.READ_PLATFORM}),
    UserRole.OPERATOR: frozenset(
        {
            Permission.READ_PLATFORM,
            Permission.TEST_CAMERAS,
            Permission.MANAGE_ALERTS,
            Permission.REVIEW_PLATES,
            Permission.REVIEW_IDENTITIES,
        }
    ),
    UserRole.ADMIN: frozenset(Permission),
}


class RBACAuthorizer:
    def allows(self, principal: Principal, permission: Permission) -> bool:
        return permission in _ROLE_PERMISSIONS[principal.role]


class StaticApiKeyAuthenticator:
    """Authenticate high-entropy Bearer keys against configured SHA-256 verifiers."""

    def __init__(self, config: AuthConfig) -> None:
        self._minimum_length = config.minimum_token_length
        self._principals = tuple(
            (
                principal.key_sha256.get_secret_value(),
                Principal(
                    id=principal.id,
                    display_name=principal.display_name or principal.id,
                    role=UserRole(principal.role),
                    authentication_method=AuthenticationMethod.API_KEY,
                ),
            )
            for principal in config.principals
            if principal.enabled
        )

    async def authenticate(self, bearer_token: str) -> Principal | None:
        if (
            len(bearer_token) < self._minimum_length
            or len(bearer_token) > 512
            or any(char.isspace() for char in bearer_token)
        ):
            return None
        candidate = hashlib.sha256(bearer_token.encode("utf-8")).hexdigest()
        matched: Principal | None = None
        for configured_hash, principal in self._principals:
            if hmac.compare_digest(candidate, configured_hash):
                matched = principal
        return matched


class DevelopmentAuthenticator:
    def __init__(self) -> None:
        self.principal = Principal(
            id="development-admin",
            display_name="Development Admin",
            role=UserRole.ADMIN,
            authentication_method=AuthenticationMethod.DEVELOPMENT,
        )

    async def authenticate(self, _bearer_token: str) -> Principal:
        return self.principal
