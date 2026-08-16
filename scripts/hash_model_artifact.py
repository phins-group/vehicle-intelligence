#!/usr/bin/env python3
"""Print the verified SHA-256 digest for a local model file or directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from vehicle_intelligence.model_artifact import sha256_directory, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    if path.is_file():
        digest = sha256_file(path)
    elif path.is_dir():
        digest = sha256_directory(path)
    else:
        parser.error(f"model artifact does not exist: {path}")
    print(digest)


if __name__ == "__main__":
    main()
