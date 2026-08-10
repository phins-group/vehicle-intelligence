"""Credential-free ONVIF discovery value objects."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class OnvifDiscoveredDevice:
    endpoint_reference: str
    xaddrs: tuple[str, ...]
    types: tuple[str, ...]
    scopes: tuple[str, ...]
    discovered_at: datetime
    remote_address: str | None = None
    name: str | None = None
    hardware: str | None = None
    locations: tuple[str, ...] = ()
    metadata_version: int | None = None

    def __post_init__(self) -> None:
        if not self.endpoint_reference.strip() or len(self.endpoint_reference) > 2048:
            raise ValueError("ONVIF endpoint reference is invalid")
        if not self.xaddrs or len(self.xaddrs) > 32:
            raise ValueError("ONVIF discovery requires one to 32 service addresses")
        for address in self.xaddrs:
            parsed = urlsplit(address)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or len(address) > 2048
            ):
                raise ValueError("ONVIF service address is invalid")
        if len(self.types) > 64 or len(self.scopes) > 256 or len(self.locations) > 32:
            raise ValueError("ONVIF discovery metadata exceeds domain limits")
        if any(len(value) > 2048 for value in (*self.types, *self.scopes, *self.locations)):
            raise ValueError("ONVIF discovery metadata value is too long")
        if self.discovered_at.tzinfo is None:
            raise ValueError("ONVIF discovery timestamp must be timezone-aware")
        if self.metadata_version is not None and self.metadata_version < 0:
            raise ValueError("ONVIF metadata version cannot be negative")
        if self.name is not None and (not self.name.strip() or len(self.name) > 256):
            raise ValueError("ONVIF device name is invalid")
        if self.hardware is not None and (not self.hardware.strip() or len(self.hardware) > 256):
            raise ValueError("ONVIF hardware name is invalid")
