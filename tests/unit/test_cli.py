from argparse import ArgumentTypeError, Namespace

import pytest

from vehicle_intelligence.config import load_settings
from vehicle_intelligence.exceptions import ConfigurationError
from vehicle_intelligence.interfaces.cli import (
    apply_overrides,
    build_parser,
    parse_aware_datetime,
)


def arguments(**updates) -> Namespace:
    values = {
        "camera_id": None,
        "camera_name": None,
        "fps_limit": None,
        "vehicle_model": None,
        "plate_model": None,
        "plate_only": None,
        "device": None,
        "ocr_device": None,
        "output": None,
        "storage": None,
        "mongo": None,
    }
    values.update(updates)
    return Namespace(**values)


def test_command_line_overrides_are_revalidated() -> None:
    with pytest.raises(ConfigurationError, match="fps_limit"):
        apply_overrides(load_settings(), arguments(fps_limit=-1.0))


def test_video_start_time_requires_timezone() -> None:
    with pytest.raises(ArgumentTypeError, match="timezone"):
        parse_aware_datetime("2026-08-08T20:30:00")

    parsed = parse_aware_datetime("2026-08-08T20:30:00+07:00")
    assert parsed.utcoffset() is not None


def test_plate_only_flag_enables_full_frame_plate_pipeline() -> None:
    parsed = build_parser().parse_args(["sample.mp4", "--plate-only"])

    settings = apply_overrides(load_settings(), parsed)

    assert settings.vision.plate_only


def test_plate_only_override_can_be_disabled_explicitly() -> None:
    base = load_settings()
    configured = base.model_copy(
        update={"vision": base.vision.model_copy(update={"plate_only": True})}
    )
    parsed = build_parser().parse_args(["sample.mp4", "--no-plate-only"])

    settings = apply_overrides(configured, parsed)

    assert not settings.vision.plate_only
