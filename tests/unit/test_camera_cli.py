import signal
from argparse import Namespace

import pytest

from vehicle_intelligence.exceptions import ConfigurationError
from vehicle_intelligence.interfaces.camera_cli import CameraShutdown, resolve_rtsp_url


class StoppableSource:
    def __init__(self) -> None:
        self.stop_requests = 0

    def request_stop(self) -> None:
        self.stop_requests += 1


def test_rtsp_url_can_be_loaded_from_environment_without_exposing_value(monkeypatch) -> None:
    monkeypatch.setenv("GATE_RTSP_URL", "rtsp://operator:secret@camera.local/live")
    value = resolve_rtsp_url(Namespace(rtsp=None, rtsp_env="GATE_RTSP_URL"))

    assert value.get_secret_value().endswith("@camera.local/live")
    assert "secret" not in str(value)


@pytest.mark.parametrize(
    "value",
    ("http://camera.local/live", "rtsp:///missing-host", "not-a-url"),
)
def test_rtsp_url_rejects_invalid_sources(value: str) -> None:
    with pytest.raises(ConfigurationError, match="RTSP source"):
        resolve_rtsp_url(Namespace(rtsp=value, rtsp_env=None))


def test_rtsp_environment_variable_must_exist(monkeypatch) -> None:
    monkeypatch.delenv("MISSING_RTSP_URL", raising=False)
    with pytest.raises(ConfigurationError, match="MISSING_RTSP_URL"):
        resolve_rtsp_url(Namespace(rtsp=None, rtsp_env="MISSING_RTSP_URL"))


def test_first_signal_requests_graceful_stop_and_second_interrupts() -> None:
    source = StoppableSource()
    shutdown = CameraShutdown(source)

    shutdown.handle(signal.SIGINT, None)

    assert shutdown.exit_code == 130
    assert source.stop_requests == 1
    with pytest.raises(KeyboardInterrupt):
        shutdown.handle(signal.SIGINT, None)
    assert source.stop_requests == 1
