"""Immutable COCO dataset builder with group-aware, leakage-safe splitting."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.exceptions import DetectorCorpusError, DetectorDatasetError
from vehicle_intelligence.training.config import DetectorDatasetConfig
from vehicle_intelligence.training.corpus import verify_plate_corpus
from vehicle_intelligence.training.domain import DatasetSplit, DetectorSample
from vehicle_intelligence.training.first_party import verify_first_party_detector_source

_EXPORT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VALIDATION, DatasetSplit.TEST)


@dataclass(frozen=True, slots=True)
class DetectorDatasetBuildResult:
    export_id: str
    directory: Path
    manifest_sha256: str
    sample_count: int
    annotation_count: int
    split_counts: dict[str, int]
    reused: bool = False


class DetectorDatasetBuilder:
    def __init__(
        self,
        config: DetectorDatasetConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._config = config
        self._clock = clock
        self._source = config.source_directory.expanduser().resolve()
        self._output_root = config.output_directory.expanduser().resolve()

    def build(self, export_id: str) -> DetectorDatasetBuildResult:
        if not _EXPORT_ID.fullmatch(export_id):
            raise DetectorDatasetError("detector dataset export id is not path-safe")
        target = _safe_target(self._output_root, export_id)
        if target.exists():
            manifest, digest = verify_detector_dataset(target)
            return _result(target, manifest, digest, reused=True)
        if not self._source.is_dir():
            raise DetectorDatasetError(
                f"detector dataset source directory is missing: {self._source}"
            )

        annotations_path = self._annotations_path()
        samples, source_digest = _load_samples(annotations_path, self._config)
        assignments = _assign_splits(samples, self._config)
        _validate_split_population(assignments, self._config)

        self._output_root.mkdir(parents=True, exist_ok=True)
        temporary = self._output_root / f".{export_id}.tmp-{uuid.uuid4().hex}"
        now = self._clock()
        if now.tzinfo is None:
            raise DetectorDatasetError("detector dataset build clock must be timezone-aware")
        now = now.astimezone(UTC)
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary,
                export_id,
                samples,
                assignments,
                source_digest,
                now,
            )
            manifest_bytes = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "manifest.json", manifest_bytes)
            digest = _sha256(manifest_bytes)
            if target.exists():
                raise DetectorDatasetError("detector dataset target already exists")
            temporary.replace(target)
            return _result(target, manifest, digest)
        except DetectorDatasetError:
            _remove_tree(temporary, self._output_root)
            raise
        except Exception as exc:
            _remove_tree(temporary, self._output_root)
            raise DetectorDatasetError("cannot build detector dataset") from exc

    def _annotations_path(self) -> Path:
        configured = self._config.annotations_file.expanduser()
        path = (
            configured.resolve()
            if configured.is_absolute()
            else (self._source / configured).resolve()
        )
        if not path.is_relative_to(self._source) or not path.is_file():
            raise DetectorDatasetError("detector annotations file is missing or outside source")
        return path

    def _materialize(
        self,
        temporary: Path,
        export_id: str,
        samples: list[DetectorSample],
        assignments: dict[str, DatasetSplit],
        source_digest: str,
        now: datetime,
    ) -> dict[str, Any]:
        source_evidence = _source_evidence(self._source, self._config.role.value)
        acceptance_eligible = all(
            sample.attributes.get("acceptanceEligible") is not False
            for sample in samples
        )
        categories = [
            {"id": index, "name": class_name, "supercategory": self._config.role.value}
            for index, class_name in enumerate(self._config.classes, start=1)
        ]
        category_ids = {item["name"]: item["id"] for item in categories}
        coco: dict[DatasetSplit, dict[str, Any]] = {
            split: {
                "info": {
                    "description": f"{self._config.role.value} detector {split.value} split",
                    "version": export_id,
                    "date_created": _timestamp(now),
                },
                "licenses": [],
                "images": [],
                "annotations": [],
                "categories": categories,
            }
            for split in _SPLITS
        }
        split_counts: Counter[str] = Counter()
        group_ids: dict[DatasetSplit, set[str]] = defaultdict(set)
        category_counts: Counter[str] = Counter()
        files: list[dict[str, Any]] = []
        seen_image_digests: dict[str, str] = {}
        annotation_id = 1

        for image_id, sample in enumerate(
            sorted(samples, key=lambda item: item.sample_id), start=1
        ):
            split = assignments[sample.group_id]
            source_path = _safe_source_image(self._source, sample.image_path)
            data = source_path.read_bytes()
            if not data or len(data) > self._config.maximum_image_bytes:
                raise DetectorDatasetError(f"image size is invalid for sample {sample.sample_id}")
            digest = _sha256(data)
            previous_group = seen_image_digests.get(digest)
            if previous_group is not None and previous_group != sample.group_id:
                raise DetectorDatasetError("identical image bytes occur in multiple dataset groups")
            seen_image_digests[digest] = sample.group_id
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise DetectorDatasetError(f"image cannot be decoded for sample {sample.sample_id}")
            height, width = image.shape[:2]
            if width * height > self._config.maximum_image_pixels:
                raise DetectorDatasetError(
                    f"image dimensions exceed limit for sample {sample.sample_id}"
                )
            _validate_annotations(sample, width, height, category_ids)

            suffix = source_path.suffix.lower().replace(".jpeg", ".jpg")
            filename = f"{hashlib.sha256(sample.sample_id.encode()).hexdigest()[:24]}{suffix}"
            relative = PurePosixPath("images", split.value, filename)
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, data)
            files.append({"path": str(relative), "sha256": digest, "size": len(data)})

            coco_image = {
                "id": image_id,
                "file_name": str(relative),
                "width": width,
                "height": height,
                "sample_id": sample.sample_id,
                "group_id": sample.group_id,
                "camera_id": sample.camera_id,
                "captured_at": _timestamp(sample.captured_at.astimezone(UTC)),
                "attributes": sample.attributes,
            }
            coco[split]["images"].append(coco_image)
            for annotation in sample.annotations:
                bbox = list(annotation.bbox.as_xywh())
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_ids[annotation.class_name],
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "attributes": annotation.attributes,
                }
                if annotation.polygon:
                    coco_annotation["segmentation"] = [
                        [
                            coordinate
                            for point in annotation.polygon
                            for coordinate in (point.x, point.y)
                        ]
                    ]
                coco[split]["annotations"].append(coco_annotation)
                annotation_id += 1
                category_counts[annotation.class_name] += 1
            split_counts[split.value] += 1
            group_ids[split].add(sample.group_id)

        annotation_counts: dict[str, int] = {}
        for split in _SPLITS:
            relative = PurePosixPath("annotations", f"{split.value}.json")
            payload = _json_bytes(coco[split], pretty=False)
            destination = temporary.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            _write_new(destination, payload)
            files.append({"path": str(relative), "sha256": _sha256(payload), "size": len(payload)})
            annotation_counts[split.value] = len(coco[split]["annotations"])

        attribution = _dataset_attribution(self._source, samples)
        attribution_path = temporary / "ATTRIBUTION.csv"
        _write_new(attribution_path, attribution)
        files.append(_file_entry(attribution_path, temporary))

        if not acceptance_eligible:
            notice = _bootstrap_notice(self._source, source_evidence)
            notice_path = temporary / "BOOTSTRAP_ONLY.md"
            _write_new(notice_path, notice)
            files.append(_file_entry(notice_path, temporary))

        readme_path = temporary / "README.md"
        _write_new(
            readme_path,
            _dataset_card(
                export_id=export_id,
                role=self._config.role.value,
                classes=self._config.classes,
                sample_count=len(samples),
                annotation_count=sum(annotation_counts.values()),
                acceptance_eligible=acceptance_eligible,
                source=source_evidence,
            ),
        )
        files.append(_file_entry(readme_path, temporary))

        manifest = {
            "schemaVersion": 1,
            "type": "DETECTOR_COCO",
            "exportId": export_id,
            "role": self._config.role.value,
            "acceptanceEligible": acceptance_eligible,
            "releaseEligible": (
                bool(source_evidence.get("releaseEligible", acceptance_eligible))
                if source_evidence is not None
                else acceptance_eligible
            ),
            "distributionEligible": (
                bool(source_evidence.get("distributionEligible", True))
                if source_evidence is not None
                else True
            ),
            "licenseStatus": (
                str(source_evidence.get("licenseReviewStatus"))
                if source_evidence is not None
                else "UNSPECIFIED"
            ),
            "createdAt": _timestamp(now),
            "classes": list(self._config.classes),
            "sampleCount": len(samples),
            "annotationCount": sum(annotation_counts.values()),
            "negativeSampleCount": sum(not sample.annotations for sample in samples),
            "splitCounts": dict(sorted(split_counts.items())),
            "splitAnnotationCounts": annotation_counts,
            "splitGroupCounts": {split.value: len(group_ids[split]) for split in _SPLITS},
            "categoryCounts": dict(sorted(category_counts.items())),
            "splitStrategy": {
                "type": "EXPLICIT_OR_GROUP_HASH",
                "seed": self._config.split.seed,
                "ratios": {
                    "train": self._config.split.train,
                    "validation": self._config.split.validation,
                    "test": self._config.split.test,
                },
            },
            "sourceAnnotationsSha256": source_digest,
            "files": sorted(files, key=lambda item: item["path"]),
        }
        if source_evidence is not None:
            manifest["source"] = source_evidence
        return manifest


def verify_detector_dataset(directory: Path) -> tuple[dict[str, Any], str]:
    """Verify all bytes and the COCO/group contract before train or evaluation."""

    root = directory.expanduser().resolve()
    if not _EXPORT_ID.fullmatch(root.name):
        raise DetectorDatasetError("detector dataset directory name is invalid")
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.stat().st_size > 100_000_000:
        raise DetectorDatasetError("detector dataset manifest is missing or oversized")
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("detector dataset manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != "DETECTOR_COCO"
        or manifest.get("exportId") != root.name
        or manifest.get("role") not in {"vehicle", "plate"}
        or not isinstance(manifest.get("acceptanceEligible"), bool)
        or not isinstance(manifest.get("classes"), list)
        or not isinstance(manifest.get("files"), list)
        or len(manifest["files"]) > 1_000_000
    ):
        raise DetectorDatasetError("detector dataset manifest contract is invalid")

    recorded_paths: set[str] = set()
    for item in manifest["files"]:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise DetectorDatasetError("detector dataset file entry is invalid")
        path = _safe_child(root, item["path"])
        if item["path"] in recorded_paths:
            raise DetectorDatasetError("detector dataset manifest contains duplicate files")
        recorded_paths.add(item["path"])
        if not path.is_file() or path.stat().st_size != int(item.get("size", -1)):
            raise DetectorDatasetError("detector dataset file size verification failed")
        if _sha256(path.read_bytes()) != item.get("sha256"):
            raise DetectorDatasetError("detector dataset checksum verification failed")

    groups_by_split: dict[str, set[str]] = {}
    image_paths: set[str] = set()
    total_samples = 0
    total_annotations = 0
    for split in _SPLITS:
        relative = f"annotations/{split.value}.json"
        if relative not in recorded_paths:
            raise DetectorDatasetError("detector dataset is missing a COCO split")
        document = _read_json(_safe_child(root, relative), "COCO annotation")
        images = document.get("images")
        annotations = document.get("annotations")
        categories = document.get("categories")
        if (
            not isinstance(images, list)
            or not isinstance(annotations, list)
            or not isinstance(categories, list)
        ):
            raise DetectorDatasetError("COCO detector document contract is invalid")
        category_names = [
            category.get("name") for category in categories if isinstance(category, dict)
        ]
        if category_names != manifest["classes"]:
            raise DetectorDatasetError("COCO categories do not match manifest class order")
        image_ids = {image.get("id") for image in images if isinstance(image, dict)}
        if len(image_ids) != len(images) or None in image_ids:
            raise DetectorDatasetError("COCO image ids must be unique")
        category_ids = {category.get("id") for category in categories if isinstance(category, dict)}
        groups: set[str] = set()
        for image in images:
            file_name = image.get("file_name")
            group_id = image.get("group_id")
            if not isinstance(file_name, str) or file_name not in recorded_paths:
                raise DetectorDatasetError("COCO image does not reference a verified file")
            if file_name in image_paths:
                raise DetectorDatasetError("COCO image is referenced by multiple splits")
            if not isinstance(group_id, str) or not group_id:
                raise DetectorDatasetError("COCO image is missing group_id")
            image_paths.add(file_name)
            groups.add(group_id)
        for annotation in annotations:
            if (
                not isinstance(annotation, dict)
                or annotation.get("image_id") not in image_ids
                or annotation.get("category_id") not in category_ids
            ):
                raise DetectorDatasetError("COCO annotation references are invalid")
        groups_by_split[split.value] = groups
        total_samples += len(images)
        total_annotations += len(annotations)

    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            if groups_by_split[left.value] & groups_by_split[right.value]:
                raise DetectorDatasetError("detector dataset group leakage detected")
    if total_samples != int(manifest.get("sampleCount", -1)):
        raise DetectorDatasetError("detector dataset sample count does not match manifest")
    if total_annotations != int(manifest.get("annotationCount", -1)):
        raise DetectorDatasetError("detector dataset annotation count does not match manifest")
    return manifest, _sha256(raw)


def _load_samples(
    path: Path,
    config: DetectorDatasetConfig,
) -> tuple[list[DetectorSample], str]:
    raw = path.read_bytes()
    samples: list[DetectorSample] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        if len(line) > config.maximum_annotation_line_bytes:
            raise DetectorDatasetError(f"annotation line {line_number} exceeds byte limit")
        if len(samples) >= config.maximum_samples:
            raise DetectorDatasetError("detector dataset exceeds configured sample limit")
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorDatasetError(f"annotation line {line_number} is invalid") from exc
        if sample.sample_id in seen_ids:
            raise DetectorDatasetError(f"duplicate detector sample id: {sample.sample_id}")
        seen_ids.add(sample.sample_id)
        samples.append(sample)
    if not samples:
        raise DetectorDatasetError("detector annotations contain no samples")
    return samples, _sha256(raw)


def _assign_splits(
    samples: list[DetectorSample],
    config: DetectorDatasetConfig,
) -> dict[str, DatasetSplit]:
    explicit: dict[str, DatasetSplit] = {}
    for sample in samples:
        if sample.split is None:
            continue
        existing = explicit.get(sample.group_id)
        if existing is not None and existing is not sample.split:
            raise DetectorDatasetError("one detector group declares multiple explicit splits")
        explicit[sample.group_id] = sample.split
    assignments: dict[str, DatasetSplit] = {}
    for group_id in sorted({sample.group_id for sample in samples}):
        if group_id in explicit:
            assignments[group_id] = explicit[group_id]
            continue
        digest = hashlib.sha256(f"{config.split.seed}:{group_id}".encode()).digest()
        value = int.from_bytes(digest[:8], "big") / 2**64
        if value < config.split.train:
            assignments[group_id] = DatasetSplit.TRAIN
        elif value < config.split.train + config.split.validation:
            assignments[group_id] = DatasetSplit.VALIDATION
        else:
            assignments[group_id] = DatasetSplit.TEST
    return assignments


def _validate_split_population(
    assignments: dict[str, DatasetSplit],
    config: DetectorDatasetConfig,
) -> None:
    if not config.split.require_non_empty:
        return
    populated = set(assignments.values())
    missing = [split.value for split in _SPLITS if split not in populated]
    if missing:
        raise DetectorDatasetError(f"detector dataset has empty group splits: {', '.join(missing)}")


def _validate_annotations(
    sample: DetectorSample,
    width: int,
    height: int,
    category_ids: dict[str, int],
) -> None:
    for annotation in sample.annotations:
        if annotation.class_name not in category_ids:
            raise DetectorDatasetError(
                f"unsupported class {annotation.class_name!r} in sample {sample.sample_id}"
            )
        bbox = annotation.bbox
        if bbox.x + bbox.width > width + 1e-6 or bbox.y + bbox.height > height + 1e-6:
            raise DetectorDatasetError(f"bounding box exceeds image for sample {sample.sample_id}")
        if any(
            point.x > width + 1e-6 or point.y > height + 1e-6
            for point in annotation.polygon
        ):
            raise DetectorDatasetError(f"polygon exceeds image for sample {sample.sample_id}")


def _safe_source_image(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DetectorDatasetError("detector image is missing or outside source directory")
    return path


def _safe_target(root: Path, name: str) -> Path:
    target = (root / name).resolve()
    if target == root or not target.is_relative_to(root):
        raise DetectorDatasetError("detector dataset target escapes output root")
    return target


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("detector dataset manifest path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("detector dataset manifest path escapes root")
    return path


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError(f"{description} is invalid") from exc
    if not isinstance(value, dict):
        raise DetectorDatasetError(f"{description} root must be an object")
    return value


def _source_evidence(root: Path, role: str) -> dict[str, Any] | None:
    corpus_path = root / "corpus-manifest.json"
    if corpus_path.exists():
        return _corpus_source_evidence(corpus_path, role)
    path = root / "source-manifest.json"
    if not path.exists():
        return None
    document = _read_json(path, "bootstrap source manifest")
    if document.get("type") == "FIRST_PARTY_DETECTOR_SOURCE":
        return _first_party_source_evidence(root, role)
    source = document.get("source")
    if (
        document.get("schemaVersion") != 1
        or document.get("type") != "DETECTOR_BOOTSTRAP_SOURCE"
        or document.get("role") != role
        or document.get("acceptanceEligible") is not False
        or not isinstance(source, dict)
    ):
        raise DetectorDatasetError("bootstrap source manifest contract is invalid")
    required = {
        "source_id",
        "dataset_url",
        "revision",
        "annotation_license",
        "image_license",
        "license_review_status",
    }
    if any(not isinstance(source.get(field), str) or not source[field] for field in required):
        raise DetectorDatasetError("bootstrap source provenance is incomplete")
    return {
        "type": "BOOTSTRAP_SOURCE",
        "id": source["source_id"],
        "url": source["dataset_url"],
        "revision": source["revision"],
        "annotationLicense": source["annotation_license"],
        "imageLicense": source["image_license"],
        "licenseReviewStatus": source["license_review_status"],
        "sourceManifestSha256": _sha256(path.read_bytes()),
    }


def _first_party_source_evidence(root: Path, role: str) -> dict[str, Any]:
    manifest, digest = verify_first_party_detector_source(root)
    if manifest.get("role") != role:
        raise DetectorDatasetError("first-party source role does not match dataset role")
    return {
        "type": "FIRST_PARTY_SOURCE",
        "id": manifest["sourceId"],
        "ownerNamespace": manifest["ownerNamespace"],
        "founderId": manifest["founderId"],
        "collectionMethod": manifest["collectionMethod"],
        "rightsAssertion": manifest["rightsAssertion"],
        "privacyClassification": manifest["privacyClassification"],
        "licenseReviewStatus": manifest["licenseStatus"],
        "releaseEligible": manifest["releaseEligible"],
        "distributionEligible": manifest["distributionEligible"],
        "sourceManifestSha256": digest,
    }


def _corpus_source_evidence(path: Path, role: str) -> dict[str, Any]:
    try:
        document, manifest_digest = verify_plate_corpus(path.parent)
    except DetectorCorpusError as exc:
        raise DetectorDatasetError("detector corpus integrity verification failed") from exc
    compilation = document.get("compilation")
    sources = document.get("sources")
    if (
        document.get("schemaVersion") != 1
        or document.get("type") != "DETECTOR_CORPUS_SOURCE"
        or document.get("role") != role
        or not isinstance(document.get("corpusId"), str)
        or not isinstance(compilation, dict)
        or not isinstance(sources, list)
        or not all(isinstance(source, dict) for source in sources)
        or not isinstance(document.get("licenseStatus"), str)
        or not isinstance(document.get("releaseEligible"), bool)
        or not isinstance(document.get("distributionEligible"), bool)
    ):
        raise DetectorDatasetError("detector corpus manifest contract is invalid")
    return {
        "type": "CURATED_CORPUS",
        "id": document["corpusId"],
        "ownerNamespace": compilation.get("ownerNamespace"),
        "founderId": compilation.get("founderId"),
        "sourceOwnershipClaimed": compilation.get("sourceOwnershipClaimed"),
        "licenseReviewStatus": document["licenseStatus"],
        "releaseEligible": document["releaseEligible"],
        "distributionEligible": document["distributionEligible"],
        "sources": sources,
        "corpusManifestSha256": manifest_digest,
    }


def _dataset_attribution(root: Path, samples: list[DetectorSample]) -> bytes:
    source_path = root / "ATTRIBUTION.csv"
    if source_path.exists():
        if not source_path.is_file() or source_path.stat().st_size > 100_000_000:
            raise DetectorDatasetError("source attribution file is invalid")
        raw = source_path.read_bytes()
        try:
            rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        except (UnicodeDecodeError, csv.Error) as exc:
            raise DetectorDatasetError("source attribution CSV is invalid") from exc
        sample_ids = [str(row.get("sample_id", "")) for row in rows]
        expected = {sample.sample_id for sample in samples}
        if len(sample_ids) != len(set(sample_ids)) or set(sample_ids) != expected:
            raise DetectorDatasetError("source attribution does not match detector samples")
        return raw

    fields = (
        "sample_id",
        "source_dataset",
        "source_revision",
        "license",
        "author",
        "landing_url",
    )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for sample in sorted(samples, key=lambda item: item.sample_id):
        attributes = sample.attributes
        writer.writerow(
            {
                "sample_id": sample.sample_id,
                "source_dataset": _attribute_text(attributes.get("sourceDataset")),
                "source_revision": _attribute_text(attributes.get("sourceRevision")),
                "license": _attribute_text(attributes.get("sourceLicense")),
                "author": _attribute_text(attributes.get("sourceAuthor")),
                "landing_url": _attribute_text(attributes.get("sourceLandingUrl")),
            }
        )
    return stream.getvalue().encode()


def _bootstrap_notice(root: Path, source: dict[str, Any] | None) -> bytes:
    existing = root / "BOOTSTRAP_ONLY.md"
    if existing.exists():
        if not existing.is_file() or existing.stat().st_size > 1_000_000:
            raise DetectorDatasetError("bootstrap notice is invalid")
        return existing.read_bytes()
    source_reference = source.get("url") or source.get("id") if source is not None else None
    source_line = f"\nSource: {source_reference}\n" if source_reference else ""
    return (
        "# Bootstrap-only detector dataset\n\n"
        "This dataset is not eligible as warehouse acceptance-test or release "
        "evidence. Complete legal and data-governance review before commercial "
        f"training or redistribution.\n{source_line}"
    ).encode()


def _dataset_card(
    *,
    export_id: str,
    role: str,
    classes: tuple[str, ...],
    sample_count: int,
    annotation_count: int,
    acceptance_eligible: bool,
    source: dict[str, Any] | None,
) -> bytes:
    warning = (
        "**BOOTSTRAP ONLY — not acceptance-test or release evidence.**"
        if not acceptance_eligible
        else "Acceptance eligibility is recorded in `manifest.json`."
    )
    source_section = "No external source metadata is declared."
    if source is not None and source.get("type") == "CURATED_CORPUS":
        source_lines = [
            f"- `{_markdown_text(item.get('id', 'unknown'))}` — "
            f"license `{_markdown_text(item.get('license', item.get('imageLicense', 'UNKNOWN')))}`"
            for item in source.get("sources", [])
            if isinstance(item, dict)
        ]
        source_section = "\n".join(
            (
                f"- Corpus: `{_markdown_text(source['id'])}`",
                f"- Compilation owner: `{_markdown_text(source.get('ownerNamespace'))}`",
                f"- Founder/steward: `{_markdown_text(source.get('founderId'))}`",
                f"- License review: `{_markdown_text(source['licenseReviewStatus'])}`",
                f"- Release eligible: `{bool(source.get('releaseEligible'))}`",
                f"- Distribution eligible: `{bool(source.get('distributionEligible'))}`",
                "",
                "Underlying sources:",
                *source_lines,
            )
        )
    elif source is not None and source.get("type") == "FIRST_PARTY_SOURCE":
        source_section = "\n".join(
            (
                f"- Source: `{_markdown_text(source['id'])}`",
                f"- Owner namespace: `{_markdown_text(source['ownerNamespace'])}`",
                f"- Founder/steward: `{_markdown_text(source['founderId'])}`",
                f"- Collection: `{_markdown_text(source['collectionMethod'])}`",
                f"- Rights assertion: `{_markdown_text(source['rightsAssertion'])}`",
                f"- License: `{_markdown_text(source['licenseReviewStatus'])}`",
                f"- Privacy: `{_markdown_text(source['privacyClassification'])}`",
                f"- Release eligible: `{bool(source['releaseEligible'])}`",
                f"- Distribution eligible: `{bool(source['distributionEligible'])}`",
            )
        )
    elif source is not None:
        source_section = "\n".join(
            (
                f"- Dataset: [{_markdown_text(source['id'])}]({_markdown_url(source['url'])})",
                f"- Pinned revision: `{_markdown_text(source['revision'])}`",
                f"- Annotation license: `{_markdown_text(source['annotationLicense'])}`",
                f"- Image license: `{_markdown_text(source['imageLicense'])}`",
                f"- License review: `{_markdown_text(source['licenseReviewStatus'])}`",
            )
        )
    rendered = f"""---
license: other
task_categories:
- object-detection
---

# {_markdown_text(export_id)}

Private `{_markdown_text(role)}` detector dataset generated by Vehicle Intelligence.

{warning}

- Samples: {sample_count}
- Annotations: {annotation_count}
- Classes: {", ".join(f'`{_markdown_text(item)}`' for item in classes)}

## Source and license

{source_section}

`license: other` is intentional: this repository contains data governed by the
source licenses above. The project does not relicense third-party images under
the source-code license. See `ATTRIBUTION.csv`, `manifest.json`, and, when
present, `BOOTSTRAP_ONLY.md` before using or redistributing the dataset.
"""
    return rendered.encode()


def _attribute_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _markdown_text(value: object) -> str:
    return str(value).replace("`", "'").replace("\r", " ").replace("\n", " ")


def _markdown_url(value: object) -> str:
    return str(value).replace("(", "%28").replace(")", "%29").replace(" ", "%20")


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256(data),
        "size": len(data),
    }


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_tree(target: Path, root: Path) -> None:
    resolved = target.resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise DetectorDatasetError("refusing to remove unsafe detector dataset path")
    if resolved.exists():
        shutil.rmtree(resolved)


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (rendered + "\n").encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> DetectorDatasetBuildResult:
    return DetectorDatasetBuildResult(
        export_id=str(manifest["exportId"]),
        directory=directory,
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        split_counts={str(key): int(value) for key, value in manifest["splitCounts"].items()},
        reused=reused,
    )
