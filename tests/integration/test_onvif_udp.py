from __future__ import annotations

import asyncio
import socket

from vehicle_intelligence.config import OnvifDiscoveryConfig
from vehicle_intelligence.infrastructure.vision.onvif_discovery import (
    WSDiscoveryOnvifProvider,
)


def probe_response() -> bytes:
    return b"""<?xml version="1.0"?>
    <s:Envelope xmlns:s="http://www.w3.org/2003/05/soap-envelope"
      xmlns:d="http://schemas.xmlsoap.org/ws/2005/04/discovery"
      xmlns:a="http://schemas.xmlsoap.org/ws/2004/08/addressing">
      <s:Body><d:ProbeMatches><d:ProbeMatch>
        <a:EndpointReference><a:Address>urn:uuid:camera-01</a:Address></a:EndpointReference>
        <d:Types>tds:Device dn:NetworkVideoTransmitter</d:Types>
        <d:Scopes>onvif://www.onvif.org/name/Main%20Gate</d:Scopes>
        <d:XAddrs>http://192.0.2.10/onvif/device_service</d:XAddrs>
        <d:MetadataVersion>7</d:MetadataVersion>
      </d:ProbeMatch></d:ProbeMatches></s:Body>
    </s:Envelope>"""


async def test_ws_discovery_sends_probe_and_receives_real_udp_response() -> None:
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.setblocking(False)
    port = server.getsockname()[1]
    received: list[bytes] = []

    async def respond() -> None:
        loop = asyncio.get_running_loop()
        payload, sender = await asyncio.wait_for(loop.sock_recvfrom(server, 65_535), 1)
        received.append(payload)
        await loop.sock_sendto(server, probe_response(), sender)

    responder = asyncio.create_task(respond())
    provider = WSDiscoveryOnvifProvider(
        OnvifDiscoveryConfig(timeout_seconds=0.2, probe_retries=1),
        destination=("127.0.0.1", port),
    )
    try:
        devices = await provider.discover()
        await responder
    finally:
        server.close()

    assert len(devices) == 1
    assert devices[0].endpoint_reference == "urn:uuid:camera-01"
    assert devices[0].remote_address == "127.0.0.1"
    assert received
    assert b"Probe" in received[0]
    assert b"tds:Device" in received[0]
