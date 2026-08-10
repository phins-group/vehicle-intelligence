"""OIDC JWT authentication with bounded claims and cached remote JWKS."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol

import jwt

from vehicle_intelligence.config import OIDCConfig
from vehicle_intelligence.domain import AuthenticationMethod, Principal, UserRole

logger = logging.getLogger(__name__)


class SigningKeyProvider(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class OIDCAuthenticator:
    """Validate signed bearer JWTs without trusting token-provided algorithms."""

    def __init__(
        self,
        config: OIDCConfig,
        signing_keys: SigningKeyProvider | None = None,
    ) -> None:
        self._config = config
        self._signing_keys = signing_keys or jwt.PyJWKClient(
            config.jwks_url,
            cache_keys=True,
            max_cached_keys=16,
            cache_jwk_set=True,
            lifespan=config.jwks_cache_seconds,
            timeout=config.jwks_timeout_seconds,
        )

    async def authenticate(self, bearer_token: str) -> Principal | None:
        if (
            not bearer_token
            or len(bearer_token) > self._config.maximum_token_length
            or any(char.isspace() for char in bearer_token)
        ):
            return None
        try:
            signing_key = await asyncio.to_thread(
                self._signing_keys.get_signing_key_from_jwt,
                bearer_token,
            )
            claims = jwt.decode(
                bearer_token,
                signing_key.key,
                algorithms=list(self._config.algorithms),
                audience=list(self._config.audiences),
                issuer=self._config.issuer,
                leeway=self._config.leeway_seconds,
                options={"require": ["exp", "sub"]},
            )
            return self._principal(claims)
        except (jwt.PyJWTError, ValueError, TypeError, KeyError):
            logger.info("OIDC bearer token rejected")
            return None
        except Exception:
            # JWKS network/parsing failures are intentionally fail-closed.
            logger.warning("OIDC signing-key resolution failed", exc_info=True)
            return None

    def _principal(self, claims: dict[str, Any]) -> Principal | None:
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 128:
            return None
        display_value = claims.get(self._config.name_claim, subject)
        if not isinstance(display_value, str) or not display_value.strip():
            display_value = subject
        display_name = display_value.strip()
        if len(display_name) > 256:
            return None

        raw_roles = claims.get(self._config.roles_claim, ())
        if isinstance(raw_roles, str):
            role_values = (raw_roles,)
        elif isinstance(raw_roles, list) and all(isinstance(item, str) for item in raw_roles):
            role_values = tuple(raw_roles[:100])
        else:
            return None
        mapped = {
            UserRole(self._config.role_mapping[item])
            for item in role_values
            if item in self._config.role_mapping
        }
        if not mapped:
            return None
        precedence = (UserRole.ADMIN, UserRole.OPERATOR, UserRole.VIEWER)
        role = next(item for item in precedence if item in mapped)
        return Principal(
            id=subject.strip(),
            display_name=display_name,
            role=role,
            authentication_method=AuthenticationMethod.OIDC,
        )
