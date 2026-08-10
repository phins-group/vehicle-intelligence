from __future__ import annotations

from datetime import UTC, datetime

from vehicle_intelligence.application.discovery import OnvifDiscoveryService
from vehicle_intelligence.domain import OnvifDiscoveredDevice
from vehicle_intelligence.infrastructure.vision.onvif_discovery import parse_probe_matches


def probe_response(*, endpoint: str = "urn:uuid:camera-01", xaddr: str | None = None) -> bytes:
    service = xaddr or "http://192.0.2.10/onvif/device_service"
    return f"""<?xml version="1.0"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
      xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
      xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing">
      <s:Body><d:ProbeMatches><d:ProbeMatch>
        <a:EndpointReference><a:Address>{endpoint}</a:Address></a:EndpointReference>
        <d:Types>tds:Device dn:NetworkVideoTransmitter</d:Types>
        <d:Scopes>onvif://www.onvif.org/name/Main%20Gate
          onvif://www.onvif.org/hardware/IPC-42
          onvif://www.onvif.org/location/Factory/Gate-A</d:Scopes>
        <d:XAddrs>{service}</d:XAddrs><d:MetadataVersion>7</d:MetadataVersion>
      </d:ProbeMatch></d:ProbeMatches></s:Body>
    </s:Envelope>""".encode()


def device(endpoint: str, name: str, xaddrs: tuple[str, ...]) -> OnvifDiscoveredDevice:
    return OnvifDiscoveredDevice(
        endpoint_reference=endpoint,
        xaddrs=xaddrs,
        types=("tds:Device",),
        scopes=(),
        name=name,
        discovered_at=datetime(2026, 8, 9, tzinfo=UTC),
    )


class FakeProvider:
    async def discover(self):
        return [
            device("urn:uuid:b", "Zulu", ("http://192.0.2.2/onvif",)),
            device("urn:uuid:a", "Alpha", ("http://192.0.2.1/onvif",)),
            device(
                "URN:UUID:A",
                "Duplicate",
                ("http://192.0.2.1/onvif", "https://192.0.2.1/onvif"),
            ),
        ]


def test_probe_parser_extracts_safe_onvif_metadata() -> None:
    parsed = parse_probe_matches(probe_response(), "192.0.2.10")

    assert len(parsed) == 1
    result = parsed[0]
    assert result.endpoint_reference == "urn:uuid:camera-01"
    assert result.xaddrs == ("http://192.0.2.10/onvif/device_service",)
    assert result.name == "Main Gate"
    assert result.hardware == "IPC-42"
    assert result.locations == ("Factory/Gate-A",)
    assert result.metadata_version == 7
    assert result.remote_address == "192.0.2.10"


def test_probe_parser_rejects_malformed_entities_and_credentialed_xaddr() -> None:
    assert parse_probe_matches(b"<!DOCTYPE x [<!ENTITY x 'bad'>]><x>&x;</x>", None) == []
    assert parse_probe_matches(b"not xml", None) == []
    assert (
        parse_probe_matches(
            probe_response(xaddr="http://admin:secret@192.0.2.10/onvif"),
            "192.0.2.10",
        )
        == []
    )


async def test_discovery_service_deduplicates_sorts_caps_and_stamps_results() -> None:
    now = datetime(2026, 8, 10, 1, 2, 3, tzinfo=UTC)
    service = OnvifDiscoveryService(FakeProvider(), maximum_results=1, clock=lambda: now)

    results = await service.discover()

    assert len(results) == 1
    assert results[0].endpoint_reference.casefold() == "urn:uuid:a"
    assert len(results[0].xaddrs) == 2
    assert results[0].discovered_at == now
