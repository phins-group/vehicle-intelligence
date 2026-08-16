"""Secure, provenance-preserving import of pinned Roboflow plate archives."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import uuid
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import cv2
import numpy as np
import yaml
from pydantic import ValidationError

from vehicle_intelligence.exceptions import DetectorCorpusError
from vehicle_intelligence.training.domain import DetectorSample

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_ROBOFLOW_SUFFIX = re.compile(
    r"_(?:jpg|jpeg|png)\.rf\.[0-9a-f]{32}$",
    re.IGNORECASE,
)
_SOURCE_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,95}$")
_MAX_ARCHIVE_FILES = 100_000
_MAX_ARCHIVE_BYTES = 4_000_000_000
_MAX_FILE_BYTES = 25_000_000
_ATTRIBUTION_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_revision",
    "license",
    "author",
    "landing_url",
    "source_original_ids",
    "image_sha256",
)


@dataclass(frozen=True, slots=True)
class RoboflowArchiveSpec:
    source_id: str
    title: str
    author: str
    dataset_url: str
    workspace: str
    project: str
    version: int
    archive_sha256: str
    exported_at: datetime
    expected_images: int
    task: Literal["detection", "classification"]
    class_names: tuple[str, ...]
    class_mapping: dict[int, str]
    classification_mapping: dict[str, str] = field(default_factory=dict)
    augmented: bool = False

    def __post_init__(self) -> None:
        if not _SOURCE_ID.fullmatch(self.source_id):
            raise ValueError("Roboflow source id is invalid")
        if self.exported_at.tzinfo is None:
            raise ValueError("Roboflow export timestamp must be timezone-aware")
        if self.task == "detection" and not self.class_mapping:
            raise ValueError("detection source requires a class mapping")
        if self.task == "classification" and not self.classification_mapping:
            raise ValueError("classification source requires a folder mapping")


ROBOFLOW_PLATE_ARCHIVES: tuple[RoboflowArchiveSpec, ...] = (
    RoboflowArchiveSpec(
        source_id="roboflow-yolov8-vit-folder-v2",
        title="YOLOv8 and Vision transformer",
        author="ComputerVision Project",
        dataset_url=(
            "https://universe.roboflow.com/computervision-project/yolov8-and-vision-transformer"
        ),
        workspace="computervision-project",
        project="yolov8-and-vision-transformer",
        version=2,
        archive_sha256="92bddf04861d42047a2efec93fd4b0ac0e9648559c6652a76def0f7215fae8e6",
        exported_at=datetime(2025, 3, 15, 12, 17, tzinfo=UTC),
        expected_images=7_953,
        task="classification",
        class_names=("Bien_So_Xe_May", "Car_long"),
        class_mapping={},
        classification_mapping={
            "Bien_So_Xe_May": "motorcycle_plate_context",
            "Car_long": "car_long_plate_context",
        },
        augmented=True,
    ),
    RoboflowArchiveSpec(
        source_id="roboflow-traffic-violation-v3",
        title="traffic_violation",
        author="TrafficManagement",
        dataset_url=(
            "https://universe.roboflow.com/trafficmanagement/traffic_violation-2nycm/dataset/3"
        ),
        workspace="trafficmanagement",
        project="traffic_violation-2nycm",
        version=3,
        archive_sha256="36cc0d6f5bb821cebe5d3ca04463171128f45008caf011422c8f19fe868e7076",
        exported_at=datetime(2026, 8, 10, 17, 41, tzinfo=UTC),
        expected_images=18_560,
        task="detection",
        class_names=("0", "plate"),
        # A visual audit confirmed that both raw classes enclose plates.
        class_mapping={0: "license_plate", 1: "license_plate"},
        augmented=True,
    ),
    RoboflowArchiveSpec(
        source_id="roboflow-vietnamese-license-plate-v1",
        title="vietnamese license plate",
        author="school",
        dataset_url=(
            "https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0/dataset/1"
        ),
        workspace="school-fuhih",
        project="vietnamese-license-plate-tptd0",
        version=1,
        archive_sha256="626c7403285814a3246def9e69c094f2162f4426fef712f2329daaa13bb5c032",
        exported_at=datetime(2024, 10, 10, 17, 3, tzinfo=UTC),
        expected_images=8_357,
        task="detection",
        class_names=("0",),
        class_mapping={0: "license_plate"},
    ),
    RoboflowArchiveSpec(
        source_id="roboflow-license-plate-detection-v1",
        title="License Plate Detection",
        author="cao phong",
        dataset_url=(
            "https://universe.roboflow.com/cao-phong-3qbun/license-plate-detection-dhfxl/dataset/1"
        ),
        workspace="cao-phong-3qbun",
        project="license-plate-detection-dhfxl",
        version=1,
        archive_sha256="f75d98a2f81811a35e858a2b3ea114b65f2286590efd384cab6524f0fa57310c",
        exported_at=datetime(2025, 9, 28, 13, 52, tzinfo=UTC),
        expected_images=2_555,
        task="detection",
        class_names=("0",),
        class_mapping={0: "license_plate"},
    ),
)


@dataclass(frozen=True, slots=True)
class RoboflowImportResult:
    source_id: str
    task: str
    directory: Path
    manifest_sha256: str
    source_image_count: int
    canonical_image_count: int
    annotation_count: int
    negative_sample_count: int
    duplicate_images_merged: int
    reused: bool = False


@dataclass(slots=True)
class _DetectionState:
    digest: str
    image_path: str
    group_id: str
    lineage: str
    original_ids: list[str]
    original_splits: set[str]
    annotations: list[dict[str, Any]]


@dataclass(slots=True)
class _ClassificationState:
    digest: str
    image_path: str
    group_id: str
    lineage: str
    label: str
    original_ids: list[str]
    original_splits: set[str]


class RoboflowPlateArchiveImporter:
    """Convert one pinned archive into a canonical PHINS source artifact."""

    def __init__(
        self,
        spec: RoboflowArchiveSpec,
        *,
        owner_namespace: str,
        founder_id: str,
        detection_output_root: Path,
        auxiliary_output_root: Path,
    ) -> None:
        self._spec = spec
        self._owner = owner_namespace
        self._founder = founder_id
        base = detection_output_root if spec.task == "detection" else auxiliary_output_root
        self._output_root = base.expanduser().resolve()

    def build(self, archive: Path) -> RoboflowImportResult:
        source_archive = archive.expanduser().resolve()
        if not source_archive.is_file():
            raise DetectorCorpusError("Roboflow archive does not exist")
        archive_digest = _sha256_file(source_archive)
        if archive_digest != self._spec.archive_sha256:
            raise DetectorCorpusError("Roboflow archive checksum does not match pinned source")
        target = _safe_target(self._output_root, self._spec.source_id)
        if target.exists():
            manifest, digest = verify_roboflow_source(target)
            return _result(target, manifest, digest, reused=True)

        self._output_root.mkdir(parents=True, exist_ok=True)
        temporary = self._output_root / f".{self._spec.source_id}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            with zipfile.ZipFile(source_archive) as zipped:
                members = _validated_members(zipped, self._spec.task)
                _validate_source_metadata(zipped, members, self._spec)
                if self._spec.task == "detection":
                    manifest = self._build_detection(
                        zipped,
                        members,
                        temporary,
                        archive_digest,
                    )
                else:
                    manifest = self._build_classification(
                        zipped,
                        members,
                        temporary,
                        archive_digest,
                    )
            manifest_bytes = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_bytes)
            digest = _sha256(manifest_bytes)
            temporary.replace(target)
            return _result(target, manifest, digest)
        except DetectorCorpusError:
            _remove_tree(temporary, self._output_root)
            raise
        except (OSError, zipfile.BadZipFile, UnicodeDecodeError, yaml.YAMLError) as exc:
            _remove_tree(temporary, self._output_root)
            raise DetectorCorpusError("cannot import Roboflow plate archive") from exc
        except Exception as exc:
            _remove_tree(temporary, self._output_root)
            raise DetectorCorpusError("unexpected Roboflow plate import failure") from exc

    def _build_detection(
        self,
        zipped: zipfile.ZipFile,
        members: dict[str, zipfile.ZipInfo],
        temporary: Path,
        archive_digest: str,
    ) -> dict[str, Any]:
        image_members, label_members = _detection_pairs(members)
        if len(image_members) != self._spec.expected_images:
            raise DetectorCorpusError("Roboflow detection image count changed")
        states: dict[str, _DetectionState] = {}
        files: list[dict[str, Any]] = []
        raw_class_counts: Counter[int] = Counter()
        split_counts: Counter[str] = Counter()
        source_annotation_count = 0
        for key in sorted(image_members):
            split, stem = key
            image_info = image_members[key]
            image_data = zipped.read(image_info)
            image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise DetectorCorpusError("Roboflow source image cannot be decoded")
            height, width = image.shape[:2]
            label_info = label_members[key]
            annotations = _parse_yolo_boxes(
                zipped.read(label_info),
                width,
                height,
                self._spec,
                raw_class_counts,
            )
            source_annotation_count += len(annotations)
            split_counts[split] += 1
            digest = _sha256(image_data)
            lineage = _source_lineage(stem)
            state = states.get(digest)
            if state is None:
                image_path = _write_image(temporary, files, image_data, digest, image_info.filename)
                state = _DetectionState(
                    digest=digest,
                    image_path=image_path,
                    group_id=_lineage_group_id(lineage),
                    lineage=lineage,
                    original_ids=[],
                    original_splits=set(),
                    annotations=[],
                )
                states[digest] = state
            elif state.lineage != lineage:
                state.group_id = _digest_group_id(digest)
            state.original_ids.append(image_info.filename)
            state.original_splits.add(split)
            state.annotations.extend(annotations)

        annotation_lines: list[bytes] = []
        provenance_lines: list[bytes] = []
        attribution_rows: list[dict[str, str]] = []
        annotation_count = 0
        negative_count = 0
        for state in sorted(states.values(), key=lambda item: item.digest):
            annotations = _unique_annotations(state.annotations)
            annotation_count += len(annotations)
            negative_count += not annotations
            sample_id = f"phins-rfplate-{state.digest[:24]}"
            attributes = self._sample_attributes(state.lineage, state.original_splits)
            attributes.update(
                {
                    "sourceOriginalCount": len(state.original_ids),
                    "sourceAugmented": self._spec.augmented,
                    "negativeSample": not annotations,
                }
            )
            sample = DetectorSample.model_validate(
                {
                    "sampleId": sample_id,
                    "imagePath": state.image_path,
                    "groupId": state.group_id,
                    "cameraId": "external-roboflow",
                    "capturedAt": self._spec.exported_at,
                    "split": None,
                    "attributes": attributes,
                    "annotations": annotations,
                }
            )
            annotation_lines.append(_model_json_line(sample))
            provenance_lines.append(
                _json_line(
                    {
                        "sampleId": sample_id,
                        "imageSha256": state.digest,
                        "sourceLineage": state.lineage,
                        "sourceOriginalIds": sorted(state.original_ids),
                        "sourceOriginalSplits": sorted(state.original_splits),
                    }
                )
            )
            attribution_rows.append(self._attribution_row(sample_id, state))

        metadata = {
            "annotations.jsonl": b"".join(annotation_lines),
            "ATTRIBUTION.csv": _attribution_csv(attribution_rows),
            "PROVENANCE.jsonl": b"".join(provenance_lines),
            "SOURCE_CARD.md": self._source_card(detection=True).encode(),
        }
        _write_metadata(temporary, files, metadata)
        return self._manifest(
            archive_digest=archive_digest,
            files=files,
            source_image_count=len(image_members),
            canonical_image_count=len(states),
            annotation_count=annotation_count,
            negative_sample_count=negative_count,
            statistics={
                "sourceAnnotationCount": source_annotation_count,
                "exactDuplicateImagesMerged": len(image_members) - len(states),
                "sourceSplitCounts": dict(sorted(split_counts.items())),
                "rawClassCounts": {
                    str(key): value for key, value in sorted(raw_class_counts.items())
                },
                "classMapping": {
                    str(key): value for key, value in sorted(self._spec.class_mapping.items())
                },
                "lineageCount": len({state.lineage for state in states.values()}),
            },
        )

    def _build_classification(
        self,
        zipped: zipfile.ZipFile,
        members: dict[str, zipfile.ZipInfo],
        temporary: Path,
        archive_digest: str,
    ) -> dict[str, Any]:
        image_members = _classification_images(members, self._spec)
        if len(image_members) != self._spec.expected_images:
            raise DetectorCorpusError("Roboflow classification image count changed")
        states: dict[str, _ClassificationState] = {}
        files: list[dict[str, Any]] = []
        label_counts: Counter[str] = Counter()
        split_counts: Counter[str] = Counter()
        for split, raw_label, info in image_members:
            image_data = zipped.read(info)
            image = cv2.imdecode(np.frombuffer(image_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise DetectorCorpusError("Roboflow classification image cannot be decoded")
            label = self._spec.classification_mapping[raw_label]
            digest = _sha256(image_data)
            lineage = _source_lineage(PurePosixPath(info.filename).stem)
            state = states.get(digest)
            if state is None:
                image_path = _write_image(temporary, files, image_data, digest, info.filename)
                state = _ClassificationState(
                    digest=digest,
                    image_path=image_path,
                    group_id=_lineage_group_id(lineage),
                    lineage=lineage,
                    label=label,
                    original_ids=[],
                    original_splits=set(),
                )
                states[digest] = state
            elif state.label != label:
                raise DetectorCorpusError("exact classification duplicate has conflicting labels")
            elif state.lineage != lineage:
                state.group_id = _digest_group_id(digest)
            state.original_ids.append(info.filename)
            state.original_splits.add(split)
            label_counts[label] += 1
            split_counts[split] += 1

        sample_lines: list[bytes] = []
        provenance_lines: list[bytes] = []
        attribution_rows: list[dict[str, str]] = []
        canonical_label_counts: Counter[str] = Counter()
        for state in sorted(states.values(), key=lambda item: item.digest):
            sample_id = f"phins-platecontext-{state.digest[:24]}"
            canonical_label_counts[state.label] += 1
            sample_lines.append(
                _json_line(
                    {
                        "schemaVersion": 1,
                        "sampleId": sample_id,
                        "imagePath": state.image_path,
                        "groupId": state.group_id,
                        "label": state.label,
                        "capturedAt": _timestamp(self._spec.exported_at),
                        "sourceLineage": state.lineage,
                        "sourceOriginalSplits": sorted(state.original_splits),
                    }
                )
            )
            provenance_lines.append(
                _json_line(
                    {
                        "sampleId": sample_id,
                        "imageSha256": state.digest,
                        "sourceLineage": state.lineage,
                        "sourceOriginalIds": sorted(state.original_ids),
                    }
                )
            )
            attribution_rows.append(self._attribution_row(sample_id, state))
        metadata = {
            "classification-samples.jsonl": b"".join(sample_lines),
            "ATTRIBUTION.csv": _attribution_csv(attribution_rows),
            "PROVENANCE.jsonl": b"".join(provenance_lines),
            "SOURCE_CARD.md": self._source_card(detection=False).encode(),
        }
        _write_metadata(temporary, files, metadata)
        return self._manifest(
            archive_digest=archive_digest,
            files=files,
            source_image_count=len(image_members),
            canonical_image_count=len(states),
            annotation_count=0,
            negative_sample_count=0,
            statistics={
                "sourceLabelCounts": dict(sorted(label_counts.items())),
                "canonicalLabelCounts": dict(sorted(canonical_label_counts.items())),
                "exactDuplicateImagesMerged": len(image_members) - len(states),
                "sourceSplitCounts": dict(sorted(split_counts.items())),
                "lineageCount": len({state.lineage for state in states.values()}),
                "detectorEligible": False,
            },
        )

    def _sample_attributes(self, lineage: str, splits: set[str]) -> dict[str, Any]:
        return {
            "corpusOwner": self._owner,
            "corpusFounderId": self._founder,
            "sourceDataset": self._spec.source_id,
            "sourceRevision": str(self._spec.version),
            "sourceLicense": "CC-BY-4.0",
            "sourceLicenseDeclaredBy": "ROBOFLOW_DATASET_CARD",
            "sourceOriginalSplits": ",".join(sorted(splits)),
            "sourceLineage": lineage,
            "licenseReviewStatus": "DECLARED_CC_BY_4_0_REVIEW_REQUIRED",
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "corpusRemappedId": True,
        }

    def _attribution_row(
        self,
        sample_id: str,
        state: _DetectionState | _ClassificationState,
    ) -> dict[str, str]:
        return {
            "sample_id": sample_id,
            "source_dataset": self._spec.source_id,
            "source_revision": str(self._spec.version),
            "license": "CC-BY-4.0",
            "author": self._spec.author,
            "landing_url": self._spec.dataset_url,
            "source_original_ids": "|".join(sorted(state.original_ids)),
            "image_sha256": state.digest,
        }

    def _manifest(
        self,
        *,
        archive_digest: str,
        files: list[dict[str, Any]],
        source_image_count: int,
        canonical_image_count: int,
        annotation_count: int,
        negative_sample_count: int,
        statistics: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "schemaVersion": 1,
            "type": (
                "DETECTOR_CANONICAL_SOURCE"
                if self._spec.task == "detection"
                else "AUXILIARY_CLASSIFICATION_SOURCE"
            ),
            "sourceId": self._spec.source_id,
            "task": self._spec.task,
            "role": "plate",
            "compilation": {
                "ownerNamespace": self._owner,
                "founderId": self._founder,
                "sourceOwnershipClaimed": False,
            },
            "source": {
                "source_id": self._spec.source_id,
                "dataset_url": self._spec.dataset_url,
                "revision": str(self._spec.version),
                "annotation_license": "CC-BY-4.0",
                "image_license": "CC-BY-4.0-DECLARED",
                "license_review_status": "DECLARED_CC_BY_4_0_REVIEW_REQUIRED",
                "acceptance_eligible": False,
            },
            "archiveSha256": archive_digest,
            "sourceImageCount": source_image_count,
            "sampleCount": canonical_image_count,
            "annotationCount": annotation_count,
            "negativeSampleCount": negative_sample_count,
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "licenseStatus": "DECLARED_CC_BY_4_0_REVIEW_REQUIRED",
            "statistics": statistics,
            "files": sorted(files, key=lambda item: item["path"]),
        }

    def _source_card(self, *, detection: bool) -> str:
        usage = (
            "This canonical source may feed the plate detector corpus."
            if detection
            else (
                "This is an auxiliary context-classification source. It has no bounding "
                "boxes and MUST NOT be treated as negative detector data."
            )
        )
        return (
            f"# {self._spec.source_id}\n\n"
            f"PHINS compilation owner: `{self._owner}`  \n"
            f"Founder/steward: `{self._founder}`  \n"
            f"Upstream: {self._spec.dataset_url}  \n"
            f"Upstream author: {self._spec.author}  \n"
            "Declared license: `CC BY 4.0`\n\n"
            f"{usage}\n\n"
            "PHINS canonical IDs and curation do not transfer ownership of upstream images. "
            "Keep attribution with every derived dataset/model release.\n"
        )


def import_known_roboflow_archives(
    archives: list[Path],
    *,
    owner_namespace: str,
    founder_id: str,
    detection_output_root: Path,
    auxiliary_output_root: Path,
) -> list[RoboflowImportResult]:
    """Match archives by SHA-256 so filenames never determine source identity."""

    by_digest = {spec.archive_sha256: spec for spec in ROBOFLOW_PLATE_ARCHIVES}
    seen: set[str] = set()
    matched: list[tuple[Path, RoboflowArchiveSpec]] = []
    for archive in archives:
        path = archive.expanduser().resolve()
        if not path.is_file():
            raise DetectorCorpusError(f"Roboflow archive does not exist: {path}")
        digest = _sha256_file(path)
        spec = by_digest.get(digest)
        if spec is None:
            raise DetectorCorpusError(f"unregistered Roboflow archive checksum: {digest}")
        if spec.source_id in seen:
            raise DetectorCorpusError(f"duplicate Roboflow source archive: {spec.source_id}")
        seen.add(spec.source_id)
        matched.append((path, spec))
    return [
        RoboflowPlateArchiveImporter(
            spec,
            owner_namespace=owner_namespace,
            founder_id=founder_id,
            detection_output_root=detection_output_root,
            auxiliary_output_root=auxiliary_output_root,
        ).build(path)
        for path, spec in matched
    ]


def verify_roboflow_source(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "source-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorCorpusError("Roboflow source manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("sourceId") != root.name
        or manifest.get("type")
        not in {"DETECTOR_CANONICAL_SOURCE", "AUXILIARY_CLASSIFICATION_SOURCE"}
        or manifest.get("acceptanceEligible") is not False
        or manifest.get("releaseEligible") is not False
        or manifest.get("distributionEligible") is not False
        or not isinstance(manifest.get("files"), list)
    ):
        raise DetectorCorpusError("Roboflow source manifest contract is invalid")
    recorded: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DetectorCorpusError("Roboflow source file entry is invalid")
        relative = entry["path"]
        path = _safe_child(root, relative)
        if relative in recorded or not path.is_file():
            raise DetectorCorpusError("Roboflow source file is missing or duplicated")
        recorded.add(relative)
        data = path.read_bytes()
        if len(data) != int(entry.get("size", -1)) or _sha256(data) != entry.get("sha256"):
            raise DetectorCorpusError("Roboflow source checksum verification failed")
    sample_file = (
        "annotations.jsonl"
        if manifest["type"] == "DETECTOR_CANONICAL_SOURCE"
        else "classification-samples.jsonl"
    )
    required = {sample_file, "ATTRIBUTION.csv", "PROVENANCE.jsonl", "SOURCE_CARD.md"}
    if not required.issubset(recorded):
        raise DetectorCorpusError("Roboflow source evidence files are missing")
    sample_count = _verify_sample_file(root, manifest, sample_file, recorded)
    if sample_count != int(manifest.get("sampleCount", -1)):
        raise DetectorCorpusError("Roboflow source sample count does not match manifest")
    return manifest, _sha256(raw)


def _verify_sample_file(
    root: Path,
    manifest: dict[str, Any],
    sample_file: str,
    recorded: set[str],
) -> int:
    sample_ids: set[str] = set()
    for line in (root / sample_file).read_bytes().splitlines():
        try:
            if manifest["type"] == "DETECTOR_CANONICAL_SOURCE":
                sample = DetectorSample.model_validate_json(line)
                sample_id = sample.sample_id
                image_path = sample.image_path
            else:
                document = json.loads(line)
                sample_id = document["sampleId"]
                image_path = document["imagePath"]
                if not isinstance(document.get("label"), str):
                    raise ValueError("classification label is invalid")
        except (
            ValidationError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            ValueError,
        ) as exc:
            raise DetectorCorpusError("Roboflow canonical sample is invalid") from exc
        if sample_id in sample_ids or image_path not in recorded:
            raise DetectorCorpusError("Roboflow canonical sample identity is invalid")
        sample_ids.add(sample_id)
    return len(sample_ids)


def _validated_members(
    zipped: zipfile.ZipFile,
    task: str,
) -> dict[str, zipfile.ZipInfo]:
    infos = zipped.infolist()
    if not 1 <= len(infos) <= _MAX_ARCHIVE_FILES:
        raise DetectorCorpusError("Roboflow archive file count is outside safe limits")
    if sum(info.file_size for info in infos) > _MAX_ARCHIVE_BYTES:
        raise DetectorCorpusError("Roboflow archive uncompressed size is outside safe limits")
    grouped: dict[str, list[zipfile.ZipInfo]] = {}
    for info in infos:
        path = PurePosixPath(info.filename)
        if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in info.filename:
            raise DetectorCorpusError("Roboflow archive contains an unsafe path")
        mode = info.external_attr >> 16
        if info.flag_bits & 1 or stat.S_ISLNK(mode):
            raise DetectorCorpusError("encrypted or symlink Roboflow entries are unsupported")
        if info.is_dir():
            continue
        if info.file_size > _MAX_FILE_BYTES or not _allowed_member(path, task):
            raise DetectorCorpusError("Roboflow archive contains an unexpected file")
        grouped.setdefault(info.filename, []).append(info)
    result: dict[str, zipfile.ZipInfo] = {}
    for name, duplicates in grouped.items():
        if len(duplicates) > 1:
            digests = {_sha256(zipped.read(info)) for info in duplicates}
            if name != "data.yaml" or len(digests) != 1:
                raise DetectorCorpusError("Roboflow archive contains conflicting duplicate paths")
        result[name] = duplicates[0]
    for required in ("README.dataset.txt", "README.roboflow.txt"):
        if required not in result:
            raise DetectorCorpusError("Roboflow archive source metadata is missing")
    if task == "detection" and "data.yaml" not in result:
        raise DetectorCorpusError("Roboflow detection archive data.yaml is missing")
    return result


def _allowed_member(path: PurePosixPath, task: str) -> bool:
    if len(path.parts) == 1:
        return path.name in {"README.dataset.txt", "README.roboflow.txt", "data.yaml"}
    if task == "detection":
        return (
            len(path.parts) == 3
            and path.parts[0] in {"train", "valid", "test"}
            and (
                (path.parts[1] == "images" and path.suffix.lower() in _IMAGE_SUFFIXES)
                or (path.parts[1] == "labels" and path.suffix.lower() == ".txt")
            )
        )
    return (
        len(path.parts) == 3
        and path.parts[0] in {"train", "valid", "test"}
        and path.suffix.lower() in _IMAGE_SUFFIXES
    )


def _validate_source_metadata(
    zipped: zipfile.ZipFile,
    members: dict[str, zipfile.ZipInfo],
    spec: RoboflowArchiveSpec,
) -> None:
    dataset_card = zipped.read(members["README.dataset.txt"]).decode("utf-8-sig")
    export_card = zipped.read(members["README.roboflow.txt"]).decode("utf-8-sig")
    if (
        spec.dataset_url.rsplit("/dataset/", 1)[0] not in dataset_card
        or "License: CC BY 4.0" not in dataset_card
        or f"includes {spec.expected_images} images" not in export_card
    ):
        raise DetectorCorpusError("Roboflow source card does not match pinned metadata")
    if spec.task != "detection":
        return
    document = yaml.safe_load(zipped.read(members["data.yaml"]))
    roboflow = document.get("roboflow") if isinstance(document, dict) else None
    names = document.get("names") if isinstance(document, dict) else None
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names)]
    if (
        not isinstance(document, dict)
        or document.get("nc") != len(spec.class_names)
        or tuple(names or ()) != spec.class_names
        or not isinstance(roboflow, dict)
        or roboflow.get("workspace") != spec.workspace
        or roboflow.get("project") != spec.project
        or int(roboflow.get("version", -1)) != spec.version
        or roboflow.get("license") != "CC BY 4.0"
    ):
        raise DetectorCorpusError("Roboflow data.yaml does not match pinned source contract")


def _detection_pairs(
    members: dict[str, zipfile.ZipInfo],
) -> tuple[
    dict[tuple[str, str], zipfile.ZipInfo],
    dict[tuple[str, str], zipfile.ZipInfo],
]:
    images: dict[tuple[str, str], zipfile.ZipInfo] = {}
    labels: dict[tuple[str, str], zipfile.ZipInfo] = {}
    for name, info in members.items():
        path = PurePosixPath(name)
        if len(path.parts) != 3:
            continue
        key = (path.parts[0], path.stem)
        target = images if path.parts[1] == "images" else labels
        if key in target:
            raise DetectorCorpusError("Roboflow archive has duplicate image/label stems")
        target[key] = info
    if set(images) != set(labels):
        raise DetectorCorpusError("Roboflow image and label pairs do not match")
    return images, labels


def _classification_images(
    members: dict[str, zipfile.ZipInfo],
    spec: RoboflowArchiveSpec,
) -> list[tuple[str, str, zipfile.ZipInfo]]:
    images: list[tuple[str, str, zipfile.ZipInfo]] = []
    for name, info in members.items():
        path = PurePosixPath(name)
        if len(path.parts) != 3:
            continue
        split, raw_label, _ = path.parts
        if raw_label not in spec.classification_mapping:
            raise DetectorCorpusError("Roboflow classification folder is unsupported")
        images.append((split, raw_label, info))
    return sorted(images, key=lambda item: item[2].filename)


def _parse_yolo_boxes(
    raw: bytes,
    width: int,
    height: int,
    spec: RoboflowArchiveSpec,
    raw_class_counts: Counter[int],
) -> list[dict[str, Any]]:
    try:
        text = raw.decode("utf-8-sig").strip()
    except UnicodeDecodeError as exc:
        raise DetectorCorpusError("Roboflow YOLO label is not UTF-8") from exc
    if not text:
        return []
    annotations: list[dict[str, Any]] = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 5:
            raise DetectorCorpusError("Roboflow YOLO annotation must be a bounding box")
        try:
            raw_class = int(fields[0])
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise DetectorCorpusError("Roboflow YOLO annotation is malformed") from exc
        if (
            raw_class not in spec.class_mapping
            or not all(0 <= value <= 1 for value in (center_x, center_y, box_width, box_height))
            or box_width <= 0
            or box_height <= 0
        ):
            raise DetectorCorpusError("Roboflow YOLO annotation is outside its contract")
        x1 = max(0.0, (center_x - box_width / 2) * width)
        y1 = max(0.0, (center_y - box_height / 2) * height)
        x2 = min(float(width), (center_x + box_width / 2) * width)
        y2 = min(float(height), (center_y + box_height / 2) * height)
        if x2 <= x1 or y2 <= y1:
            raise DetectorCorpusError("Roboflow YOLO annotation collapses after clamping")
        raw_class_counts[raw_class] += 1
        annotations.append(
            {
                "className": spec.class_mapping[raw_class],
                "bbox": {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1},
                "attributes": {
                    "rawClassId": raw_class,
                    "rawClassName": spec.class_names[raw_class],
                    "sourceFormat": "YOLOV11",
                },
            }
        )
    return annotations


def _unique_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    unique: list[dict[str, Any]] = []
    for annotation in annotations:
        bbox = annotation["bbox"]
        key = (
            annotation["className"],
            *(round(float(bbox[name]), 6) for name in ("x", "y", "width", "height")),
        )
        if key not in seen:
            seen.add(key)
            unique.append(annotation)
    return unique


def _source_lineage(stem: str) -> str:
    raw = _ROBOFLOW_SUFFIX.sub("", stem).casefold()
    normalized = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    if not normalized:
        raise DetectorCorpusError("Roboflow source lineage is empty")
    return normalized


def _lineage_group_id(lineage: str) -> str:
    return f"phins-origin:{_sha256(lineage.encode())[:24]}"


def _digest_group_id(digest: str) -> str:
    return f"phins-origin-dedup:{digest[:24]}"


def _write_image(
    temporary: Path,
    files: list[dict[str, Any]],
    data: bytes,
    digest: str,
    source_name: str,
) -> str:
    suffix = PurePosixPath(source_name).suffix.lower().replace(".jpeg", ".jpg")
    relative = f"images/{digest[:2]}/{digest}{suffix}"
    path = _safe_child(temporary, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_new(path, data)
    files.append(_file_entry(path, temporary))
    return relative


def _write_metadata(
    temporary: Path,
    files: list[dict[str, Any]],
    metadata: dict[str, bytes],
) -> None:
    for relative, data in metadata.items():
        path = _safe_child(temporary, relative)
        _write_new(path, data)
        files.append(_file_entry(path, temporary))


def _attribution_csv(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_ATTRIBUTION_FIELDS)
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))
    return stream.getvalue().encode()


def _model_json_line(sample: DetectorSample) -> bytes:
    return _json_line(sample.model_dump(mode="json", by_alias=True))


def _json_line(document: dict[str, Any]) -> bytes:
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode()


def _json_bytes(document: dict[str, Any], *, pretty: bool) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode()


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": path.relative_to(root).as_posix(), "sha256": _sha256(data), "size": len(data)}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_target(root: Path, name: str) -> Path:
    if not _SOURCE_ID.fullmatch(name):
        raise DetectorCorpusError("Roboflow source id is not path-safe")
    return _safe_child(root, name)


def _safe_child(root: Path, relative: str) -> Path:
    path = PurePosixPath(relative)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DetectorCorpusError("Roboflow source path is unsafe")
    candidate = root.joinpath(*path.parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise DetectorCorpusError("Roboflow source path escapes its root")
    return candidate


def _write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _remove_tree(path: Path, root: Path) -> None:
    if path.exists() and path.parent.resolve() == root.resolve() and path.name.startswith("."):
        shutil.rmtree(path)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> RoboflowImportResult:
    statistics = manifest.get("statistics", {})
    return RoboflowImportResult(
        source_id=str(manifest["sourceId"]),
        task=str(manifest["task"]),
        directory=directory,
        manifest_sha256=digest,
        source_image_count=int(manifest["sourceImageCount"]),
        canonical_image_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        negative_sample_count=int(manifest["negativeSampleCount"]),
        duplicate_images_merged=int(statistics.get("exactDuplicateImagesMerged", 0)),
        reused=reused,
    )
