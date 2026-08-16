#!/usr/bin/env python3
"""Fail when executable CI/container dependencies use mutable references."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SHA256_REFERENCE = re.compile(r"@sha256:[0-9a-f]{64}$")
ACTION_REFERENCE = re.compile(r"@[0-9a-f]{40}$")
DOCKER_INSTRUCTION = re.compile(r"^(ARG|FROM)\s+(.+)$", re.IGNORECASE)
DOCKER_VARIABLE = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "build",
        "datasets",
        "dist",
        "node_modules",
        "output",
        "outputs",
    }
)


def main() -> None:
    failures = [*_dockerfile_failures(), *_compose_failures(), *_workflow_failures()]
    if failures:
        raise SystemExit("\n".join(failures))


def _dockerfile_failures() -> list[str]:
    failures: list[str] = []
    for path in _source_files("Dockerfile*"):
        arguments: dict[str, str] = {}
        stages: set[str] = set()
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            line = raw_line.strip()
            instruction = DOCKER_INSTRUCTION.fullmatch(line)
            if instruction is None:
                continue
            command, value = instruction.groups()
            if command.upper() == "ARG" and "=" in value:
                name, value = value.split("=", 1)
                arguments[name] = value
                continue
            if command.upper() != "FROM":
                continue
            tokens = value.split()
            while tokens and tokens[0].startswith("--"):
                tokens.pop(0)
            reference = tokens[0] if tokens else ""
            reference = DOCKER_VARIABLE.sub(
                lambda match, arguments=arguments: arguments.get(
                    match.group("braced") or match.group("plain"), ""
                ),
                reference,
            )
            if reference.lower() not in {"scratch", *stages} and not SHA256_REFERENCE.search(
                reference
            ):
                relative_path = path.relative_to(ROOT)
                failures.append(f"{relative_path}:{line_number}: unpinned FROM {reference!r}")
            if len(tokens) >= 3 and tokens[-2].upper() == "AS":
                stages.add(tokens[-1].lower())
    return failures


def _compose_failures() -> list[str]:
    failures: list[str] = []
    compose_paths = {
        *_source_files("compose*.yml"),
        *_source_files("compose*.yaml"),
        *_source_files("docker-compose*.yml"),
        *_source_files("docker-compose*.yaml"),
    }
    for path in sorted(compose_paths):
        document = yaml.safe_load(path.read_text()) or {}
        services = document.get("services", {}) if isinstance(document, dict) else {}
        for service, config in services.items():
            reference = config.get("image") if isinstance(config, dict) else None
            if reference is not None and not SHA256_REFERENCE.search(str(reference)):
                relative_path = path.relative_to(ROOT)
                failures.append(f"{relative_path}:{service}: unpinned image {reference!r}")
    return failures


def _workflow_failures() -> list[str]:
    failures: list[str] = []
    workflow_root = ROOT / ".github" / "workflows"
    paths = [*workflow_root.glob("*.yml"), *workflow_root.glob("*.yaml")]
    for path in sorted(paths):
        for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
            stripped = raw_line.strip().removeprefix("-").lstrip()
            if not stripped.startswith("uses:"):
                continue
            reference = stripped.removeprefix("uses:").split("#", 1)[0].strip()
            if reference.startswith("./"):
                continue
            if not ACTION_REFERENCE.search(reference):
                failures.append(f"{path.relative_to(ROOT)}:{line_number}: unpinned action")
    return failures


def _source_files(pattern: str) -> list[Path]:
    paths: list[Path] = []
    for directory, child_directories, filenames in os.walk(ROOT):
        child_directories[:] = [
            name for name in child_directories if name not in EXCLUDED_DIRECTORIES
        ]
        paths.extend(
            Path(directory, filename)
            for filename in filenames
            if fnmatch.fnmatch(filename, pattern)
        )
    return sorted(paths)


if __name__ == "__main__":
    main()
