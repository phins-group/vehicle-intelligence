"""Race-safe ownership and cleanup for the local inference socket."""

from __future__ import annotations

import os
import socket
import stat
from dataclasses import dataclass
from pathlib import Path

from vehicle_intelligence.exceptions import InferenceError


@dataclass(frozen=True, slots=True)
class SocketIdentity:
    device: int
    inode: int


def prepare_socket_path(path: Path) -> None:
    """Create a private parent and remove only a provably stale socket."""
    _ensure_private_parent(path.parent)
    identity = socket_identity(path)
    if identity is None:
        return
    if _socket_accepts_connections(path):
        raise InferenceError("shared inference socket is already serving")
    unlink_owned_socket(path, identity)


def socket_identity(path: Path) -> SocketIdentity | None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise InferenceError("cannot inspect shared inference socket path") from exc
    if not stat.S_ISSOCK(details.st_mode):
        raise InferenceError("shared inference socket path is occupied by a non-socket")
    return SocketIdentity(details.st_dev, details.st_ino)


def unlink_owned_socket(path: Path, identity: SocketIdentity | None) -> bool:
    if identity is None:
        return False
    try:
        current = path.lstat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise InferenceError("cannot inspect shared inference socket during cleanup") from exc
    if (
        not stat.S_ISSOCK(current.st_mode)
        or current.st_dev != identity.device
        or current.st_ino != identity.inode
    ):
        return False
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise InferenceError("cannot remove owned shared inference socket") from exc
    return True


def _ensure_private_parent(parent: Path) -> None:
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        details = parent.lstat()
    except OSError as exc:
        raise InferenceError("cannot create shared inference socket directory") from exc
    if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
        raise InferenceError("shared inference socket directory must be a real directory")
    if details.st_uid != os.getuid():
        raise InferenceError("shared inference socket directory has a different owner")
    if stat.S_IMODE(details.st_mode) & 0o077:
        raise InferenceError("shared inference socket directory must have mode 0700 or stricter")


def _socket_accepts_connections(path: Path) -> bool:
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.2)
            probe.connect(str(path))
        return True
    except (ConnectionRefusedError, FileNotFoundError):
        return False
    except OSError as exc:
        raise InferenceError("cannot safely probe existing shared inference socket") from exc
