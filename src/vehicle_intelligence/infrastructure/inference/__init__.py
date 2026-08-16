"""Bounded local IPC for the supervisor-owned shared inference process."""

from vehicle_intelligence.infrastructure.inference.protocol import (
    INFERENCE_CAMERA_ENV,
    INFERENCE_SOCKET_ENV,
    INFERENCE_TOKEN_ENV,
)

__all__ = ["INFERENCE_CAMERA_ENV", "INFERENCE_SOCKET_ENV", "INFERENCE_TOKEN_ENV"]
