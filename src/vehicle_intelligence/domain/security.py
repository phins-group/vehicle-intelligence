from __future__ import annotations

from dataclasses import dataclass

from vehicle_intelligence.domain.enums import AuthenticationMethod, UserRole


@dataclass(frozen=True, slots=True)
class Principal:
    id: str
    display_name: str
    role: UserRole
    authentication_method: AuthenticationMethod

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.display_name.strip():
            raise ValueError("principal id and display name are required")
        if len(self.id) > 128 or len(self.display_name) > 256:
            raise ValueError("principal identity fields are too long")
