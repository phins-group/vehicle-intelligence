"""Atomic canonical source writer and verifier for bootstrap samples."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import ValidationError

from vehicle_intelligence.exceptions import SampleDataAcquisitionError
from vehicle_intelligence.training.bootstrap.domain import (
    AcquiredDetectorSample,
    BootstrapBuildResult,
    BootstrapSourceInfo,
)
from vehicle_intelligence.training.domain import DetectorRole, DetectorSample

_ATTRIBUTION_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_revision",
    "license",
    "author",
    "landing_url",
)


class BootstrapSourceWriter:
    def __init__(self, role: DetectorRole, output_directory: Path) -> None:
        self._role = role
        self._target = output_directory.expanduser().resolve()

    def write(
        self,
        source: BootstrapSourceInfo,
        samples: list[AcquiredDetectorSample],
    ) -> BootstrapBuildResult:
        if source.acceptance_eligible:
            raise SampleDataAcquisitionError("external bootstrap source cannot be acceptance data")
        if self._target.exists():
            manifest, digest = verify_bootstrap_source(self._target)
            return _result(self._target, manifest, digest, reused=True)
        if not samples:
            raise SampleDataAcquisitionError("bootstrap source contains no samples")
        self._target.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._target.parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            files: list[dict[str, Any]] = []
            records: list[bytes] = []
            annotation_count = 0
            for acquired in sorted(samples, key=lambda item: item.sample.sample_id):
                sample = acquired.sample
                if sample.attributes.get("acceptanceEligible") is not False:
                    raise SampleDataAcquisitionError(
                        "bootstrap sample must be marked acceptance-ineligible"
                    )
                image_path = _safe_child(temporary, sample.image_path)
                image_path.parent.mkdir(parents=True, exist_ok=True)
                _write_new(image_path, acquired.image_bytes)
                files.append(_file_entry(image_path, temporary))
                records.append(
                    (
                        json.dumps(
                            sample.model_dump(mode="json", by_alias=True),
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode()
                )
                annotation_count += len(sample.annotations)
            annotations_path = temporary / "annotations.jsonl"
            _write_new(annotations_path, b"".join(records))
            files.append(_file_entry(annotations_path, temporary))
            attribution_path = temporary / "ATTRIBUTION.csv"
            _write_new(attribution_path, _attribution_csv(samples))
            files.append(_file_entry(attribution_path, temporary))
            notice_path = temporary / "BOOTSTRAP_ONLY.md"
            _write_new(notice_path, _notice(source).encode())
            files.append(_file_entry(notice_path, temporary))
            manifest = {
                "schemaVersion": 1,
                "type": "DETECTOR_BOOTSTRAP_SOURCE",
                "role": self._role.value,
                "acceptanceEligible": False,
                "licenseReviewStatus": source.license_review_status,
                "source": source.model_dump(mode="json"),
                "sampleCount": len(samples),
                "annotationCount": annotation_count,
                "files": sorted(files, key=lambda item: item["path"]),
            }
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            _write_new(temporary / "source-manifest.json", manifest_bytes)
            digest = _sha256(manifest_bytes)
            temporary.replace(self._target)
            return BootstrapBuildResult(
                role=self._role,
                directory=self._target,
                manifest_sha256=digest,
                sample_count=len(samples),
                annotation_count=annotation_count,
            )
        except SampleDataAcquisitionError:
            _remove_tree(temporary, self._target.parent)
            raise
        except Exception as exc:
            _remove_tree(temporary, self._target.parent)
            raise SampleDataAcquisitionError("cannot write bootstrap detector source") from exc


def verify_bootstrap_source(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "source-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SampleDataAcquisitionError("bootstrap source manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != "DETECTOR_BOOTSTRAP_SOURCE"
        or manifest.get("role") not in {"vehicle", "plate"}
        or manifest.get("acceptanceEligible") is not False
        or manifest.get("licenseReviewStatus") != "REVIEW_REQUIRED"
        or not isinstance(manifest.get("files"), list)
    ):
        raise SampleDataAcquisitionError("bootstrap source manifest contract is invalid")
    recorded_paths: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SampleDataAcquisitionError("bootstrap source file entry is invalid")
        relative = entry["path"]
        path = _safe_child(root, relative)
        if relative in recorded_paths or not path.is_file():
            raise SampleDataAcquisitionError("bootstrap source file is missing or duplicated")
        recorded_paths.add(relative)
        if path.stat().st_size != int(entry.get("size", -1)):
            raise SampleDataAcquisitionError("bootstrap source file size verification failed")
        if _sha256(path.read_bytes()) != entry.get("sha256"):
            raise SampleDataAcquisitionError("bootstrap source checksum verification failed")
    if "annotations.jsonl" not in recorded_paths or "ATTRIBUTION.csv" not in recorded_paths:
        raise SampleDataAcquisitionError("bootstrap source evidence files are missing")
    sample_count = 0
    annotation_count = 0
    image_paths: set[str] = set()
    for line in (root / "annotations.jsonl").read_bytes().splitlines():
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise SampleDataAcquisitionError("bootstrap annotation is invalid") from exc
        if sample.attributes.get("acceptanceEligible") is not False:
            raise SampleDataAcquisitionError("bootstrap annotation is acceptance-eligible")
        if sample.image_path not in recorded_paths or sample.image_path in image_paths:
            raise SampleDataAcquisitionError("bootstrap image reference is invalid")
        image_paths.add(sample.image_path)
        sample_count += 1
        annotation_count += len(sample.annotations)
    if sample_count != int(manifest.get("sampleCount", -1)):
        raise SampleDataAcquisitionError("bootstrap sample count does not match manifest")
    if annotation_count != int(manifest.get("annotationCount", -1)):
        raise SampleDataAcquisitionError("bootstrap annotation count does not match manifest")
    return manifest, _sha256(raw)


def _attribution_csv(samples: list[AcquiredDetectorSample]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_ATTRIBUTION_FIELDS, extrasaction="raise")
    writer.writeheader()
    for acquired in sorted(samples, key=lambda item: item.sample.sample_id):
        writer.writerow(acquired.attribution)
    return stream.getvalue().encode()


def _notice(source: BootstrapSourceInfo) -> str:
    return f"""# Bootstrap-only detector samples

These externally sourced samples are for pipeline smoke tests and initial
fine-tuning experiments. They are not warehouse acceptance-test evidence and
cannot be packaged as a release candidate by this project.

- Source: {source.dataset_url}
- Revision: {source.revision}
- Annotation license: {source.annotation_license}
- Image license: {source.image_license}
- License review: {source.license_review_status}

Keep attribution and complete a legal/data-governance review before any
commercial redistribution or production training decision.
"""


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    data = path.read_bytes()
    return {"path": relative, "sha256": _sha256(data), "size": len(data)}


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise SampleDataAcquisitionError("bootstrap path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise SampleDataAcquisitionError("bootstrap path escapes target")
    return path


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_tree(target: Path, root: Path) -> None:
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise SampleDataAcquisitionError("refusing to remove unsafe bootstrap path")
    if resolved.exists():
        shutil.rmtree(resolved)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool,
) -> BootstrapBuildResult:
    return BootstrapBuildResult(
        role=DetectorRole(str(manifest["role"])),
        directory=directory,
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        reused=reused,
    )
