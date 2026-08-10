"""FastAPI authentication and RBAC dependency adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from vehicle_intelligence.application.ports import Authenticator
from vehicle_intelligence.application.security import Permission, RBACAuthorizer
from vehicle_intelligence.config import AuthConfig
from vehicle_intelligence.domain import Principal

_bearer = HTTPBearer(auto_error=False)


class APISecurity:
    def __init__(
        self,
        config: AuthConfig,
        authenticator: Authenticator,
        authorizer: RBACAuthorizer | None = None,
    ) -> None:
        self.enabled = config.enabled
        self._realm = config.realm
        self._authenticator = authenticator
        self._authorizer = authorizer or RBACAuthorizer()

    async def current_principal(
        self,
        credentials: HTTPAuthorizationCredentials | None = Security(_bearer),
    ) -> Principal:
        token = "" if credentials is None else credentials.credentials
        if credentials is not None and credentials.scheme.casefold() != "bearer":
            raise self._unauthorized()
        principal = await self.authenticate_token(token)
        if principal is None:
            raise self._unauthorized()
        return principal

    async def authenticate_token(self, token: str) -> Principal | None:
        return await self._authenticator.authenticate(token)

    def allows(self, principal: Principal, permission: Permission) -> bool:
        return self._authorizer.allows(principal, permission)

    def require(self, permission: Permission) -> Callable[..., Awaitable[Principal]]:
        async def authorize(
            principal: Principal = Depends(self.current_principal),
        ) -> Principal:
            if not self.allows(principal, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="insufficient permission",
                )
            return principal

        return authorize

    def _unauthorized(self) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing bearer token",
            headers={"WWW-Authenticate": f'Bearer realm="{self._realm}"'},
        )


class PrincipalPublic(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str
    display_name: str = Field(alias="displayName")
    role: str
    authentication_method: str = Field(alias="authenticationMethod")

    @classmethod
    def from_domain(cls, principal: Principal) -> PrincipalPublic:
        return cls(
            id=principal.id,
            displayName=principal.display_name,
            role=principal.role.value,
            authenticationMethod=principal.authentication_method.value,
        )


def build_auth_router(security: APISecurity) -> APIRouter:
    router = APIRouter(prefix="/api/auth", tags=["authentication"])
    read_access = security.require(Permission.READ_PLATFORM)

    @router.get("/me", response_model=PrincipalPublic)
    async def me(
        principal: Principal = Depends(read_access),
    ) -> PrincipalPublic:
        return PrincipalPublic.from_domain(principal)

    return router
