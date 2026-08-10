"""Bounded ONVIF WS-Discovery adapter using only the Python standard library."""

from __future__ import annotations

import asyncio
import socket
import uuid
import xml.etree.ElementTree as ET
from contextlib import suppress
from datetime import UTC, datetime
from time import monotonic
from urllib.parse import unquote

from vehicle_intelligence.config import OnvifDiscoveryConfig
from vehicle_intelligence.domain import OnvifDiscoveredDevice
from vehicle_intelligence.exceptions import CameraDiscoveryError

_DISCOVERY_NAMESPACE = "http://schemas.xmlsoap.org/ws/2005/04/discovery"
_SOAP_NAMESPACE = "http://www.w3.org/2003/05/soap-envelope"
_ADDRESSING_NAMESPACE = "http://schemas.xmlsoap.org/ws/2004/08/addressing"
_ONVIF_DEVICE_NAMESPACE = "http://www.onvif.org/ver10/device/wsdl"
_ONVIF_NETWORK_NAMESPACE = "http://www.onvif.org/ver10/network/wsdl"
_DESTINATION = ("239.255.255.250", 3702)
_SCOPE_PREFIXES = {
    "name": "onvif://www.onvif.org/name/",
    "hardware": "onvif://www.onvif.org/hardware/",
    "location": "onvif://www.onvif.org/location/",
}


class WSDiscoveryOnvifProvider:
    def __init__(
        self,
        config: OnvifDiscoveryConfig,
        destination: tuple[str, int] | None = None,
    ) -> None:
        self._config = config
        self._destination = destination or (config.multicast_address, config.port)

    async def discover(self) -> list[OnvifDiscoveredDevice]:
        return await asyncio.to_thread(self._discover_blocking)

    def _discover_blocking(self) -> list[OnvifDiscoveredDevice]:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_TTL,
                self._config.multicast_ttl,
            )
            if self._config.interface_address is not None:
                sock.setsockopt(
                    socket.IPPROTO_IP,
                    socket.IP_MULTICAST_IF,
                    socket.inet_aton(self._config.interface_address),
                )
            sock.bind((self._config.interface_address or "0.0.0.0", 0))
            return self._exchange(sock)
        except OSError as exc:
            raise CameraDiscoveryError("ONVIF discovery socket is unavailable") from exc
        finally:
            sock.close()

    def _exchange(self, sock: socket.socket) -> list[OnvifDiscoveredDevice]:
        deadline = monotonic() + self._config.timeout_seconds
        devices: dict[str, OnvifDiscoveredDevice] = {}
        response_limit = min(self._config.maximum_response_bytes + 1, 65_535)

        for attempt in range(self._config.probe_retries):
            for probe_type in ("tds:Device", "dn:NetworkVideoTransmitter"):
                payload = _probe_message(probe_type)
                try:
                    sock.sendto(payload, self._destination)
                except OSError as exc:
                    raise CameraDiscoveryError("cannot send ONVIF discovery probe") from exc

            attempts_left = self._config.probe_retries - attempt
            slice_deadline = monotonic() + max(
                0.0,
                (deadline - monotonic()) / attempts_left,
            )
            while len(devices) < self._config.maximum_results:
                remaining = min(deadline, slice_deadline) - monotonic()
                if remaining <= 0:
                    break
                try:
                    sock.settimeout(remaining)
                    data, sender = sock.recvfrom(response_limit)
                except TimeoutError:
                    break
                except OSError as exc:
                    raise CameraDiscoveryError("cannot receive ONVIF discovery response") from exc
                if len(data) > self._config.maximum_response_bytes:
                    continue
                for device in parse_probe_matches(data, str(sender[0])):
                    key = device.endpoint_reference.casefold()
                    devices[key] = _merge(devices.get(key), device)
            if monotonic() >= deadline:
                break

        return list(devices.values())


def _probe_message(probe_type: str) -> bytes:
    message_id = f"urn:uuid:{uuid.uuid4()}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<s:Envelope xmlns:s="{_SOAP_NAMESPACE}" '
        f'xmlns:a="{_ADDRESSING_NAMESPACE}" '
        f'xmlns:d="{_DISCOVERY_NAMESPACE}" '
        f'xmlns:tds="{_ONVIF_DEVICE_NAMESPACE}" '
        f'xmlns:dn="{_ONVIF_NETWORK_NAMESPACE}">'
        "<s:Header>"
        f"<a:MessageID>{message_id}</a:MessageID>"
        '<a:To s:mustUnderstand="true">'
        "urn:schemas-xmlsoap-org:ws:2005:04:discovery"
        "</a:To>"
        '<a:Action s:mustUnderstand="true">'
        "http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe"
        "</a:Action>"
        "</s:Header>"
        f"<s:Body><d:Probe><d:Types>{probe_type}</d:Types></d:Probe></s:Body>"
        "</s:Envelope>"
    ).encode()


def parse_probe_matches(payload: bytes, remote_address: str | None) -> list[OnvifDiscoveredDevice]:
    lowered = payload.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        return []
    try:
        root = ET.fromstring(payload)
    except ET.ParseError:
        return []

    now = datetime.now(UTC)
    devices: list[OnvifDiscoveredDevice] = []
    for match in (item for item in root.iter() if _local_name(item.tag) == "ProbeMatch"):
        endpoint_container = _first(match, "EndpointReference")
        endpoint = _text(endpoint_container, "Address") if endpoint_container is not None else None
        xaddrs = _words(_text(match, "XAddrs"))
        if not endpoint or not xaddrs:
            continue
        types = _words(_text(match, "Types"))
        scopes = _words(_text(match, "Scopes"))
        metadata_version = _integer(_text(match, "MetadataVersion"))
        name = _scope_value(scopes, "name")
        hardware = _scope_value(scopes, "hardware")
        locations = tuple(
            value
            for scope in scopes
            if (value := _scope_suffix(scope, _SCOPE_PREFIXES["location"])) is not None
        )
        try:
            devices.append(
                OnvifDiscoveredDevice(
                    endpoint_reference=endpoint,
                    xaddrs=tuple(dict.fromkeys(xaddrs)),
                    types=tuple(dict.fromkeys(types)),
                    scopes=tuple(dict.fromkeys(scopes)),
                    remote_address=remote_address,
                    name=name,
                    hardware=hardware,
                    locations=tuple(dict.fromkeys(locations)),
                    metadata_version=metadata_version,
                    discovered_at=now,
                )
            )
        except ValueError:
            continue
    return devices


def _merge(
    current: OnvifDiscoveredDevice | None,
    incoming: OnvifDiscoveredDevice,
) -> OnvifDiscoveredDevice:
    if current is None:
        return incoming
    return OnvifDiscoveredDevice(
        endpoint_reference=current.endpoint_reference,
        xaddrs=tuple(dict.fromkeys((*current.xaddrs, *incoming.xaddrs))),
        types=tuple(dict.fromkeys((*current.types, *incoming.types))),
        scopes=tuple(dict.fromkeys((*current.scopes, *incoming.scopes))),
        remote_address=current.remote_address or incoming.remote_address,
        name=current.name or incoming.name,
        hardware=current.hardware or incoming.hardware,
        locations=tuple(dict.fromkeys((*current.locations, *incoming.locations))),
        metadata_version=max(
            value
            for value in (current.metadata_version, incoming.metadata_version)
            if value is not None
        )
        if current.metadata_version is not None or incoming.metadata_version is not None
        else None,
        discovered_at=max(current.discovered_at, incoming.discovered_at),
    )


def _first(element: ET.Element, name: str) -> ET.Element | None:
    return next((item for item in element.iter() if _local_name(item.tag) == name), None)


def _text(element: ET.Element | None, name: str) -> str | None:
    if element is None:
        return None
    item = _first(element, name)
    if item is None or item.text is None:
        return None
    value = item.text.strip()
    return value or None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _words(value: str | None) -> tuple[str, ...]:
    return tuple(value.split()) if value else ()


def _integer(value: str | None) -> int | None:
    if value is None:
        return None
    with suppress(ValueError):
        parsed = int(value)
        return parsed if parsed >= 0 else None
    return None


def _scope_value(scopes: tuple[str, ...], kind: str) -> str | None:
    prefix = _SCOPE_PREFIXES[kind]
    return next(
        (value for scope in scopes if (value := _scope_suffix(scope, prefix)) is not None),
        None,
    )


def _scope_suffix(scope: str, prefix: str) -> str | None:
    if not scope.casefold().startswith(prefix.casefold()):
        return None
    value = unquote(scope[len(prefix) :]).strip()
    return value[:256] or None
