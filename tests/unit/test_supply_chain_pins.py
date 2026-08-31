from pathlib import Path

from scripts import check_supply_chain_pins as pins


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_nested_unpinned_dockerfile_is_reported(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    _write(tmp_path / "apps/web/Dockerfile", "FROM node:24-alpine AS build\n")

    assert pins._dockerfile_failures() == ["apps/web/Dockerfile:1: unpinned FROM 'node:24-alpine'"]


def test_nested_pinned_dockerfile_and_excluded_dependencies_pass(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    digest = "a" * 64
    _write(tmp_path / "apps/web/Dockerfile", f"FROM node:24-alpine@sha256:{digest}\n")
    _write(tmp_path / "apps/web/node_modules/example/Dockerfile", "FROM mutable:latest\n")

    assert pins._dockerfile_failures() == []


def test_dockerfile_parser_handles_case_platform_scratch_and_stage_alias(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    digest = "c" * 64
    _write(
        tmp_path / "Dockerfile",
        "\n".join(
            (
                f"from --platform=$BUILDPLATFORM node:24@sha256:{digest} AS build",
                "FROM build AS assets",
                "FROM scratch",
            )
        ),
    )

    assert pins._dockerfile_failures() == []


def test_lowercase_unpinned_dockerfile_cannot_bypass_checker(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    _write(tmp_path / "Dockerfile", "from python:3.12-slim\n")

    assert pins._dockerfile_failures() == ["Dockerfile:1: unpinned FROM 'python:3.12-slim'"]


def test_all_compose_variants_are_checked(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    digest = "b" * 64
    _write(
        tmp_path / "docker-compose.yml",
        f"services:\n  database:\n    image: mongo:8@sha256:{digest}\n",
    )
    _write(
        tmp_path / "deploy/compose.edge.yaml",
        "services:\n  broker:\n    image: redis:latest\n",
    )

    assert pins._compose_failures() == [
        "deploy/compose.edge.yaml:broker: unpinned image 'redis:latest'"
    ]


def test_named_production_image_inputs_must_fail_when_omitted(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    _write(
        tmp_path / "docker-compose.production.yml",
        "services:\n  api:\n    image: "
        "${VIP_API_IMAGE:?Set image}@sha256:${VIP_API_IMAGE_SHA256:?Set digest}\n",
    )

    assert pins._compose_failures() == []

    _write(
        tmp_path / "docker-compose.production.yml",
        "services:\n  api:\n    image: "
        "${VIP_API_IMAGE:-mutable:latest}@sha256:${VIP_API_IMAGE_SHA256:?Set digest}\n",
    )

    assert pins._compose_failures() == [
        "docker-compose.production.yml:api: unpinned image "
        "'${VIP_API_IMAGE:-mutable:latest}@sha256:${VIP_API_IMAGE_SHA256:?Set digest}'"
    ]


def test_unapproved_compose_image_input_cannot_bypass_pin_check(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    _write(
        tmp_path / "docker-compose.production.yml",
        "services:\n  database:\n    image: "
        "${DATABASE_IMAGE:?Set image}@sha256:${DATABASE_IMAGE_SHA256:?Set digest}\n",
    )

    assert pins._compose_failures() == [
        "docker-compose.production.yml:database: unpinned image "
        "'${DATABASE_IMAGE:?Set image}@sha256:${DATABASE_IMAGE_SHA256:?Set digest}'"
    ]


def test_yaml_workflow_actions_must_be_commit_pinned(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(pins, "ROOT", tmp_path)
    _write(
        tmp_path / ".github/workflows/security.yaml",
        "jobs:\n  audit:\n    steps:\n      - uses: actions/checkout@v4\n",
    )

    assert pins._workflow_failures() == [".github/workflows/security.yaml:4: unpinned action"]
