"""Founder-namespaced, provenance-preserving Vietnam plate corpus ingestion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import shutil
import stat
import statistics
import uuid
import zipfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
import yaml
from pydantic import ValidationError

from vehicle_intelligence.exceptions import DetectorCorpusError, SampleDataAcquisitionError
from vehicle_intelligence.training.bootstrap import verify_bootstrap_source
from vehicle_intelligence.training.config import DataCorpusConfig
from vehicle_intelligence.training.domain import DatasetSplit, DetectorSample
from vehicle_intelligence.training.roboflow import verify_roboflow_source

_EXPECTED_ARCHIVE_SHA256 = "0bda89d129502a4c0f05e7b693b639f655b7eda0a85d2ee6639a0227fd968865"
_SOURCE_PAGE = "https://www.kaggle.com/datasets/duydieunguyen/licenseplates"
_SOURCE_PUBLISHED_AT = datetime(2023, 7, 22, 2, 48, tzinfo=UTC)
_SOURCE_GROUP_SPLITS = {
    "Dieu": DatasetSplit.VALIDATION,
    "Hung": DatasetSplit.TRAIN,
    "Tgmt": DatasetSplit.TEST,
    "carlong": DatasetSplit.TRAIN,
    "greenpack": DatasetSplit.TRAIN,
}
_SOURCE_CLASSES = {
    0: ("BSD", "ONE_LINE"),
    1: ("BSV", "TWO_LINE"),
}
_ATTRIBUTION_FIELDS = (
    "sample_id",
    "corpus_owner",
    "source_dataset",
    "source_revision",
    "license",
    "author",
    "landing_url",
    "source_original_ids",
    "image_sha256",
)


@dataclass(frozen=True, slots=True)
class PlateCorpusBuildResult:
    corpus_id: str
    directory: Path
    manifest_sha256: str
    sample_count: int
    annotation_count: int
    duplicate_images_merged: int
    reused: bool = False


@dataclass(frozen=True, slots=True)
class _Candidate:
    bbox: tuple[float, float, float, float]
    polygon: tuple[tuple[float, float], ...]
    layout: str | None
    attributes: dict[str, str | bool | int | float | None]


@dataclass(slots=True)
class _SampleState:
    sample_id: str
    image_path: str
    image_sha256: str
    group_id: str
    captured_at: datetime
    split: DatasetSplit | None
    attributes: dict[str, str | bool | int | float | None]
    candidates: list[_Candidate] = field(default_factory=list)
    source_records: list[dict[str, str]] = field(default_factory=list)
    source_original_ids: list[str] = field(default_factory=list)


class VietnamPlateCorpusBuilder:
    """Import the pinned Kaggle polygon archive into a PHINS-owned compilation.

    PHINS identifiers describe curation and stewardship only. Source ownership,
    author, URL, revision, and unknown license remain explicit in every export.
    """

    def __init__(
        self,
        config: DataCorpusConfig,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        expected_archive_sha256: str = _EXPECTED_ARCHIVE_SHA256,
    ) -> None:
        self._config = config
        self._clock = clock
        self._expected_archive_sha256 = expected_archive_sha256
        self._root = config.plate_output_directory.expanduser().resolve()

    def build(self, archive: Path) -> PlateCorpusBuildResult:
        target = _safe_target(self._root, self._config.plate_corpus_id)
        if target.exists():
            manifest, digest = verify_plate_corpus(target)
            return _result(target, manifest, digest, reused=True)
        source_archive = archive.expanduser().resolve()
        if not source_archive.is_file():
            raise DetectorCorpusError("Vietnam plate archive does not exist")
        archive_digest = _sha256_file(source_archive)
        if archive_digest != self._expected_archive_sha256:
            raise DetectorCorpusError(
                "Vietnam plate archive hash does not match pinned Kaggle version"
            )

        self._root.mkdir(parents=True, exist_ok=True)
        temporary = self._root / f".{self._config.plate_corpus_id}.tmp-{uuid.uuid4().hex}"
        now = self._clock()
        if now.tzinfo is None:
            raise DetectorCorpusError("plate corpus build clock must be timezone-aware")
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            states: dict[str, _SampleState] = {}
            image_files: list[dict[str, Any]] = []
            rejects: list[dict[str, Any]] = []
            statistics_data: Counter[str] = Counter()
            sources = [self._kaggle_source(archive_digest)]
            with zipfile.ZipFile(source_archive) as zipped:
                self._ingest_archive(
                    zipped,
                    temporary,
                    states,
                    image_files,
                    rejects,
                    statistics_data,
                )
            for source in self._config.plate_additional_sources:
                source_metadata = self._ingest_canonical_source(
                    source.expanduser().resolve(),
                    temporary,
                    states,
                    image_files,
                    statistics_data,
                )
                if source_metadata is not None:
                    sources.append(source_metadata)
            manifest = self._write_corpus(
                temporary,
                states,
                image_files,
                rejects,
                statistics_data,
                sources,
                now.astimezone(UTC),
            )
            manifest_bytes = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "corpus-manifest.json", manifest_bytes)
            digest = _sha256(manifest_bytes)
            temporary.replace(target)
            return _result(target, manifest, digest)
        except DetectorCorpusError:
            _remove_tree(temporary, self._root)
            raise
        except (OSError, zipfile.BadZipFile, yaml.YAMLError) as exc:
            _remove_tree(temporary, self._root)
            raise DetectorCorpusError("cannot import Vietnam plate corpus") from exc
        except Exception as exc:
            _remove_tree(temporary, self._root)
            raise DetectorCorpusError("unexpected Vietnam plate corpus import failure") from exc

    def _ingest_archive(
        self,
        zipped: zipfile.ZipFile,
        temporary: Path,
        states: dict[str, _SampleState],
        image_files: list[dict[str, Any]],
        rejects: list[dict[str, Any]],
        statistics_data: Counter[str],
    ) -> None:
        infos = _validated_archive_members(zipped)
        _validate_dataset_yaml(zipped.read("dataset.yaml"))
        images: dict[tuple[str, str], str] = {}
        labels: dict[tuple[str, str], str] = {}
        groups: set[str] = set()
        for name in infos:
            parts = name.split("/")
            if len(parts) != 3:
                continue
            kind, split, filename = parts
            stem = PurePosixPath(filename).stem
            key = (split, stem)
            if kind == "images":
                images[key] = name
                groups.add(_source_group(stem))
            elif kind == "labels":
                labels[key] = name
        if set(images) != set(labels) or groups != set(_SOURCE_GROUP_SPLITS):
            raise DetectorCorpusError("archive image/label pairs or source groups changed")

        for key in sorted(images):
            original_split, stem = key
            image_name = images[key]
            label_name = labels[key]
            data = zipped.read(image_name)
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                rejects.append({"sourceOriginalId": image_name, "reason": "IMAGE_DECODE"})
                statistics_data["rejectedImages"] += 1
                continue
            height, width = image.shape[:2]
            candidates = _parse_yolo_polygons(
                zipped.read(label_name),
                width,
                height,
                label_name,
                rejects,
                statistics_data,
            )
            if not candidates:
                rejects.append({"sourceOriginalId": image_name, "reason": "NO_VALID_ANNOTATION"})
                statistics_data["rejectedImages"] += 1
                continue
            source_group = _source_group(stem)
            record = {
                "source_dataset": "kaggle-3543299",
                "source_revision": "6174984",
                "license": "UNKNOWN",
                "author": "Duy Diệu Nguyễn and collaborator",
                "landing_url": _SOURCE_PAGE,
            }
            attributes = _image_quality_attributes(image)
            attributes.update(
                {
                    "sourceDataset": "kaggle-3543299",
                    "sourceRevision": "6174984",
                    "sourceLicense": "UNKNOWN",
                    "sourceOriginalSplit": original_split,
                    "sourceSequence": source_group,
                    "licenseReviewStatus": "REVIEW_REQUIRED",
                    "acceptanceEligible": False,
                    "releaseEligible": False,
                    "distributionEligible": False,
                    "capturedAtSemantics": "SOURCE_VERSION_PUBLISHED_AT",
                }
            )
            state, duplicate = self._state_for_image(
                temporary,
                states,
                image_files,
                data=data,
                suffix=".png",
                group_id=_owned_group_id(self._config.owner_namespace, source_group),
                split=_SOURCE_GROUP_SPLITS[source_group],
                captured_at=_SOURCE_PUBLISHED_AT,
                attributes=attributes,
                source_record=record,
                source_original_id=image_name,
            )
            state.candidates.extend(candidates)
            statistics_data["sourceImages"] += 1
            statistics_data["sourceAnnotations"] += len(candidates)
            if duplicate:
                statistics_data["duplicateImagesMerged"] += 1

    def _ingest_canonical_source(
        self,
        root: Path,
        temporary: Path,
        states: dict[str, _SampleState],
        image_files: list[dict[str, Any]],
        statistics_data: Counter[str],
    ) -> dict[str, Any] | None:
        annotations_path = root / "annotations.jsonl"
        if not annotations_path.is_file():
            raise DetectorCorpusError(f"additional plate source is invalid: {root}")
        attribution = _read_source_attribution(root / "ATTRIBUTION.csv")
        source_manifest = _verified_source_manifest(root)
        source_info = source_manifest.get("source") if source_manifest is not None else None
        for line_number, line in enumerate(annotations_path.read_bytes().splitlines(), start=1):
            if not line.strip():
                continue
            try:
                sample = DetectorSample.model_validate_json(line)
            except ValidationError as exc:
                raise DetectorCorpusError(
                    f"additional source annotation line {line_number} is invalid"
                ) from exc
            source_image = _safe_source_image(root, sample.image_path)
            data = source_image.read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise DetectorCorpusError("additional source image cannot be decoded")
            source_record = attribution.get(sample.sample_id) or {
                "source_dataset": _text(sample.attributes.get("sourceDataset")),
                "source_revision": _text(sample.attributes.get("sourceRevision")),
                "license": _text(sample.attributes.get("sourceLicense")),
                "author": "UNKNOWN",
                "landing_url": "",
            }
            attributes = dict(sample.attributes)
            attributes.update(_image_quality_attributes(image))
            attributes.update(
                {
                    "acceptanceEligible": False,
                    "releaseEligible": False,
                    "corpusRemappedId": True,
                }
            )
            source_group = _owned_group_id(
                self._config.owner_namespace,
                f"additional:{sample.group_id}",
            )
            state, duplicate = self._state_for_image(
                temporary,
                states,
                image_files,
                data=data,
                suffix=source_image.suffix.lower().replace(".jpeg", ".jpg"),
                group_id=source_group,
                split=sample.split,
                captured_at=sample.captured_at.astimezone(UTC),
                attributes=attributes,
                source_record=source_record,
                source_original_id=sample.sample_id,
            )
            for annotation in sample.annotations:
                state.candidates.append(
                    _Candidate(
                        bbox=annotation.bbox.as_xywh(),
                        polygon=tuple((point.x, point.y) for point in annotation.polygon),
                        layout=_text(annotation.attributes.get("layout")) or None,
                        attributes=dict(annotation.attributes),
                    )
                )
            statistics_data["additionalSourceImages"] += 1
            statistics_data["additionalSourceAnnotations"] += len(sample.annotations)
            if duplicate:
                statistics_data["duplicateImagesMerged"] += 1
        if not isinstance(source_info, dict):
            return None
        return {
            "id": _text(source_info.get("source_id")),
            "url": _text(source_info.get("dataset_url")),
            "revision": _text(source_info.get("revision")),
            "annotationLicense": _text(source_info.get("annotation_license")),
            "imageLicense": _text(source_info.get("image_license")),
            "licenseReviewStatus": _text(source_info.get("license_review_status")),
        }

    def _state_for_image(
        self,
        temporary: Path,
        states: dict[str, _SampleState],
        image_files: list[dict[str, Any]],
        *,
        data: bytes,
        suffix: str,
        group_id: str,
        split: DatasetSplit | None,
        captured_at: datetime,
        attributes: dict[str, str | bool | int | float | None],
        source_record: dict[str, str],
        source_original_id: str,
    ) -> tuple[_SampleState, bool]:
        digest = _sha256(data)
        existing = states.get(digest)
        if existing is not None:
            existing.source_records.append(source_record)
            existing.source_original_ids.append(source_original_id)
            if existing.group_id != group_id:
                existing.group_id = _owned_group_id(self._config.owner_namespace, f"dedup:{digest}")
                existing.split = None
            return existing, True
        sample_id = f"phins-vnplate-{digest[:24]}"
        image_path = f"images/{digest[:2]}/{digest}{suffix}"
        destination = _safe_child(temporary, image_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_new(destination, data)
        image_files.append(_file_entry(destination, temporary))
        state = _SampleState(
            sample_id=sample_id,
            image_path=image_path,
            image_sha256=digest,
            group_id=group_id,
            captured_at=captured_at,
            split=split,
            attributes=attributes,
            source_records=[source_record],
            source_original_ids=[source_original_id],
        )
        states[digest] = state
        return state, False

    def _write_corpus(
        self,
        temporary: Path,
        states: dict[str, _SampleState],
        image_files: list[dict[str, Any]],
        rejects: list[dict[str, Any]],
        statistics_data: Counter[str],
        sources: list[dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        annotations: list[bytes] = []
        provenance: list[bytes] = []
        attribution_rows: list[dict[str, str]] = []
        annotation_count = 0
        split_counts: Counter[str] = Counter()
        layout_counts: Counter[str] = Counter()
        lighting_counts: Counter[str] = Counter()
        for state in sorted(states.values(), key=lambda item: item.sample_id):
            reconciled, consensus_count = _reconcile_candidates(state.candidates)
            statistics_data["consensusAnnotationsMerged"] += consensus_count
            annotation_count += len(reconciled)
            for annotation in reconciled:
                layout_counts[_text(annotation["attributes"].get("layout")) or "UNKNOWN"] += 1
            lighting_counts[_text(state.attributes.get("lighting")) or "UNKNOWN"] += 1
            split_counts[state.split.value if state.split is not None else "unassigned"] += 1
            attributes = dict(state.attributes)
            attributes.update(
                {
                    "corpusOwner": self._config.owner_namespace,
                    "corpusFounderId": self._config.founder_id,
                    "corpusId": self._config.plate_corpus_id,
                    "sourceOriginalCount": len(state.source_original_ids),
                }
            )
            sample = DetectorSample.model_validate(
                {
                    "sampleId": state.sample_id,
                    "imagePath": state.image_path,
                    "groupId": state.group_id,
                    "cameraId": "external-curated-corpus",
                    "capturedAt": state.captured_at,
                    "split": state.split,
                    "attributes": attributes,
                    "annotations": reconciled,
                }
            )
            annotations.append(
                (
                    json.dumps(
                        sample.model_dump(mode="json", by_alias=True),
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    )
                    + "\n"
                ).encode()
            )
            provenance.append(
                _json_bytes(
                    {
                        "sampleId": state.sample_id,
                        "imageSha256": state.image_sha256,
                        "sourceOriginalIds": sorted(set(state.source_original_ids)),
                        "sources": _unique_source_records(state.source_records),
                    },
                    pretty=False,
                )
            )
            attribution_rows.append(
                _attribution_row(
                    state,
                    self._config.owner_namespace,
                )
            )

        files = list(image_files)
        metadata = {
            "annotations.jsonl": b"".join(annotations),
            "ATTRIBUTION.csv": _attribution_csv(attribution_rows),
            "PROVENANCE.jsonl": b"".join(provenance),
            "REJECTS.jsonl": b"".join(_json_bytes(item, pretty=False) for item in rejects),
            "DATA_GOVERNANCE.md": _governance_notice(self._config, sources).encode(),
        }
        for relative, data in metadata.items():
            path = temporary / relative
            _write_new(path, data)
            files.append(_file_entry(path, temporary))
        return {
            "schemaVersion": 1,
            "type": "DETECTOR_CORPUS_SOURCE",
            "corpusId": self._config.plate_corpus_id,
            "role": "plate",
            "compilation": {
                "ownerNamespace": self._config.owner_namespace,
                "founderId": self._config.founder_id,
                "sampleIdScheme": "phins-vnplate-sha256-v1",
                "sourceOwnershipClaimed": False,
            },
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "licenseStatus": "REVIEW_REQUIRED_UNKNOWN_SOURCE_LICENSE",
            "createdAt": _timestamp(now),
            "sampleCount": len(states),
            "annotationCount": annotation_count,
            "splitCounts": dict(sorted(split_counts.items())),
            "layoutCounts": dict(sorted(layout_counts.items())),
            "lightingCounts": dict(sorted(lighting_counts.items())),
            "statistics": dict(sorted(statistics_data.items())),
            "sources": sources,
            "files": sorted(files, key=lambda item: item["path"]),
        }

    @staticmethod
    def _kaggle_source(archive_digest: str) -> dict[str, Any]:
        return {
            "id": "kaggle-3543299",
            "versionId": "6174984",
            "url": _SOURCE_PAGE,
            "title": "Vietnam License Plate Segment Datasets",
            "author": "Duy Diệu Nguyễn and collaborator",
            "license": "UNKNOWN",
            "licenseReviewStatus": "REVIEW_REQUIRED",
            "archiveSha256": archive_digest,
        }


def verify_plate_corpus(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "corpus-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorCorpusError("plate corpus manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != "DETECTOR_CORPUS_SOURCE"
        or manifest.get("corpusId") != root.name
        or manifest.get("role") != "plate"
        or manifest.get("acceptanceEligible") is not False
        or manifest.get("releaseEligible") is not False
        or manifest.get("distributionEligible") is not False
        or not isinstance(manifest.get("files"), list)
    ):
        raise DetectorCorpusError("plate corpus manifest contract is invalid")
    recorded: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DetectorCorpusError("plate corpus file entry is invalid")
        relative = entry["path"]
        path = _safe_child(root, relative)
        if relative in recorded or not path.is_file():
            raise DetectorCorpusError("plate corpus file is missing or duplicated")
        recorded.add(relative)
        data = path.read_bytes()
        if len(data) != int(entry.get("size", -1)) or _sha256(data) != entry.get("sha256"):
            raise DetectorCorpusError("plate corpus checksum verification failed")
    required = {
        "annotations.jsonl",
        "ATTRIBUTION.csv",
        "PROVENANCE.jsonl",
        "REJECTS.jsonl",
        "DATA_GOVERNANCE.md",
    }
    if not required.issubset(recorded):
        raise DetectorCorpusError("plate corpus evidence files are missing")
    sample_ids: set[str] = set()
    image_paths: set[str] = set()
    annotation_count = 0
    for line in (root / "annotations.jsonl").read_bytes().splitlines():
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorCorpusError("plate corpus annotation is invalid") from exc
        if (
            not sample.sample_id.startswith("phins-vnplate-")
            or sample.sample_id in sample_ids
            or sample.image_path in image_paths
            or sample.image_path not in recorded
            or sample.attributes.get("corpusOwner")
            != manifest.get("compilation", {}).get("ownerNamespace")
        ):
            raise DetectorCorpusError("plate corpus canonical identity is invalid")
        sample_ids.add(sample.sample_id)
        image_paths.add(sample.image_path)
        annotation_count += len(sample.annotations)
    if len(sample_ids) != int(manifest.get("sampleCount", -1)):
        raise DetectorCorpusError("plate corpus sample count does not match manifest")
    if annotation_count != int(manifest.get("annotationCount", -1)):
        raise DetectorCorpusError("plate corpus annotation count does not match manifest")
    attribution_ids = _read_attribution_ids(root / "ATTRIBUTION.csv")
    if attribution_ids != sample_ids:
        raise DetectorCorpusError("plate corpus attribution does not match samples")
    return manifest, _sha256(raw)


def _validated_archive_members(zipped: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = zipped.infolist()
    if not 1 <= len(members) <= 20_000:
        raise DetectorCorpusError("archive file count is outside safe limits")
    if sum(item.file_size for item in members) > 2_000_000_000:
        raise DetectorCorpusError("archive uncompressed size is outside safe limits")
    validated: dict[str, zipfile.ZipInfo] = {}
    for item in members:
        path = PurePosixPath(item.filename)
        if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in item.filename:
            raise DetectorCorpusError("archive contains an unsafe path")
        mode = item.external_attr >> 16
        if item.flag_bits & 1 or stat.S_ISLNK(mode):
            raise DetectorCorpusError("encrypted or symlink archive entries are unsupported")
        if item.is_dir():
            continue
        allowed = item.filename == "dataset.yaml" or _allowed_archive_data_path(path)
        if not allowed or item.file_size > 20_000_000:
            raise DetectorCorpusError("archive contains an unexpected or oversized file")
        if item.filename in validated:
            raise DetectorCorpusError("archive contains duplicate paths")
        validated[item.filename] = item
    if "dataset.yaml" not in validated:
        raise DetectorCorpusError("archive dataset.yaml is missing")
    return validated


def _allowed_archive_data_path(path: PurePosixPath) -> bool:
    if len(path.parts) != 3 or path.parts[1] not in {"train", "val"}:
        return False
    if path.parts[0] == "images":
        return path.suffix.lower() == ".png"
    if path.parts[0] == "labels":
        return path.suffix.lower() == ".txt"
    return False


def _validate_dataset_yaml(raw: bytes) -> None:
    try:
        document = yaml.safe_load(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise DetectorCorpusError("archive dataset.yaml is invalid") from exc
    if (
        not isinstance(document, dict)
        or document.get("nc") != 2
        or document.get("names") != ["BSD", "BSV"]
    ):
        raise DetectorCorpusError("archive class mapping changed")


def _parse_yolo_polygons(
    raw: bytes,
    width: int,
    height: int,
    source_name: str,
    rejects: list[dict[str, Any]],
    statistics_data: Counter[str],
) -> list[_Candidate]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DetectorCorpusError("archive label is not UTF-8") from exc
    candidates: list[_Candidate] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
            normalized = [float(value) for value in parts[1:]]
        except (IndexError, ValueError):
            rejects.append(
                {"sourceOriginalId": source_name, "line": line_number, "reason": "NON_NUMERIC"}
            )
            statistics_data["rejectedAnnotations"] += 1
            continue
        if class_id not in _SOURCE_CLASSES or len(normalized) < 6 or len(normalized) % 2:
            rejects.append(
                {"sourceOriginalId": source_name, "line": line_number, "reason": "SCHEMA"}
            )
            statistics_data["rejectedAnnotations"] += 1
            continue
        if any(value < -0.03 or value > 1.03 for value in normalized):
            rejects.append(
                {"sourceOriginalId": source_name, "line": line_number, "reason": "COORD_RANGE"}
            )
            statistics_data["rejectedAnnotations"] += 1
            continue
        adjusted = any(value < 0 or value > 1 for value in normalized)
        clamped = [min(1.0, max(0.0, value)) for value in normalized]
        points = tuple(
            (clamped[index] * width, clamped[index + 1] * height)
            for index in range(0, len(clamped), 2)
        )
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        if x2 - x1 < 1 or y2 - y1 < 1:
            rejects.append(
                {"sourceOriginalId": source_name, "line": line_number, "reason": "DEGENERATE"}
            )
            statistics_data["rejectedAnnotations"] += 1
            continue
        source_class, layout = _SOURCE_CLASSES[class_id]
        attributes: dict[str, str | bool | int | float | None] = {
            "layout": layout,
            "sourceClassId": class_id,
            "sourceClassName": source_class,
            "sourceVertexCount": len(points),
            "coordinateAdjusted": adjusted,
            "bboxAreaRatio": round(((x2 - x1) * (y2 - y1)) / (width * height), 8),
        }
        candidates.append(
            _Candidate(
                bbox=(x1, y1, x2 - x1, y2 - y1),
                polygon=points,
                layout=layout,
                attributes=attributes,
            )
        )
        statistics_data[f"sourceClass:{source_class}"] += 1
        if adjusted:
            statistics_data["coordinateAdjustedAnnotations"] += 1
    return candidates


def _reconcile_candidates(
    candidates: list[_Candidate],
) -> tuple[list[dict[str, Any]], int]:
    clusters: list[list[_Candidate]] = []
    for candidate in candidates:
        matched = next(
            (
                cluster
                for cluster in clusters
                if cluster[0].layout == candidate.layout
                and _bbox_iou(cluster[0].bbox, candidate.bbox) >= 0.85
            ),
            None,
        )
        if matched is None:
            clusters.append([candidate])
        else:
            matched.append(candidate)
    output: list[dict[str, Any]] = []
    merged = 0
    for cluster in clusters:
        merged += len(cluster) - 1
        chosen = cluster[0]
        polygon = chosen.polygon
        if polygon and all(len(item.polygon) == len(polygon) for item in cluster):
            polygon = tuple(
                (
                    statistics.median(item.polygon[index][0] for item in cluster),
                    statistics.median(item.polygon[index][1] for item in cluster),
                )
                for index in range(len(polygon))
            )
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            bbox = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            bbox = chosen.bbox
        attributes = dict(chosen.attributes)
        attributes["sourceConsensusCount"] = len(cluster)
        output.append(
            {
                "className": "license_plate",
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "width": bbox[2],
                    "height": bbox[3],
                },
                "polygon": [{"x": point[0], "y": point[1]} for point in polygon],
                "attributes": attributes,
            }
        )
    return output, merged


def _bbox_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x2, left_y2 = left[0] + left[2], left[1] + left[3]
    right_x2, right_y2 = right[0] + right[2], right[1] + right[3]
    width = max(0.0, min(left_x2, right_x2) - max(left[0], right[0]))
    height = max(0.0, min(left_y2, right_y2) - max(left[1], right[1]))
    intersection = width * height
    union = left[2] * left[3] + right[2] * right[3] - intersection
    return intersection / union if union > 0 else 0.0


def _image_quality_attributes(image: np.ndarray) -> dict[str, str | bool | int | float | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray) / 255.0)
    contrast = float(np.std(gray) / 128.0)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return {
        "lighting": "NIGHT" if brightness < 0.28 else "DAY",
        "lightingEstimated": True,
        "imageBrightness": round(brightness, 6),
        "imageContrast": round(min(1.0, contrast), 6),
        "imageSharpness": round(sharpness, 4),
    }


def _source_group(stem: str) -> str:
    parts = stem.rsplit("_", maxsplit=1)
    if len(parts) != 2 or not parts[0] or not parts[1].isdigit():
        raise DetectorCorpusError("archive source filename does not contain a stable sequence")
    return parts[0]


def _owned_group_id(namespace: str, source_key: str) -> str:
    digest = hashlib.sha256(source_key.encode()).hexdigest()[:20]
    return f"{namespace}:plate-sequence:{digest}"


def _read_source_attribution(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DetectorCorpusError("additional source attribution is invalid") from exc
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if not sample_id or sample_id in result:
            raise DetectorCorpusError("additional source attribution ids are invalid")
        result[sample_id] = {key: str(value or "") for key, value in row.items()}
    return result


def _read_attribution_ids(path: Path) -> set[str]:
    try:
        rows = list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8-sig"))))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise DetectorCorpusError("plate corpus attribution CSV is invalid") from exc
    ids = [str(row.get("sample_id", "")) for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise DetectorCorpusError("plate corpus attribution ids are invalid")
    return set(ids)


def _attribution_row(state: _SampleState, owner: str) -> dict[str, str]:
    records = _unique_source_records(state.source_records)
    return {
        "sample_id": state.sample_id,
        "corpus_owner": owner,
        "source_dataset": _joined(records, "source_dataset"),
        "source_revision": _joined(records, "source_revision"),
        "license": _joined(records, "license"),
        "author": _joined(records, "author"),
        "landing_url": _joined(records, "landing_url"),
        "source_original_ids": json.dumps(
            sorted(set(state.source_original_ids)), ensure_ascii=False, separators=(",", ":")
        ),
        "image_sha256": state.image_sha256,
    }


def _unique_source_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    unique: dict[str, dict[str, str]] = {}
    for record in records:
        rendered = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        unique[rendered] = record
    return [unique[key] for key in sorted(unique)]


def _joined(records: list[dict[str, str]], key: str) -> str:
    return " | ".join(sorted({record.get(key, "") for record in records if record.get(key, "")}))


def _attribution_csv(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_ATTRIBUTION_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def _governance_notice(config: DataCorpusConfig, sources: list[dict[str, Any]]) -> str:
    source_lines = "\n".join(
        f"- {item.get('id', 'unknown')}: {item.get('url', '')} "
        f"(license: {item.get('license', item.get('imageLicense', 'UNKNOWN'))})"
        for item in sources
    )
    return f"""# {config.plate_corpus_id}

Compilation namespace: `{config.owner_namespace}`
Founder/steward identifier: `{config.founder_id}`

PHINS owns the corpus organization, canonical identifiers, curation pipeline,
quality metadata, and derived manifests. PHINS does **not** claim ownership of
third-party source images. Original source identity and attribution remain in
`ATTRIBUTION.csv` and `PROVENANCE.jsonl`.

This corpus is not release, acceptance, or distribution eligible because the
Kaggle source declares an unknown license. Obtain written permission or a clear
commercial license before production training, model release, redistribution,
or upload to an external registry.

## Sources

{source_lines}
"""


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorCorpusError("additional source manifest is invalid") from exc
    if not isinstance(value, dict):
        raise DetectorCorpusError("additional source manifest must be an object")
    return value


def _verified_source_manifest(root: Path) -> dict[str, Any] | None:
    document = _optional_json(root / "source-manifest.json")
    if document is None:
        return None
    source_type = document.get("type")
    try:
        if source_type == "DETECTOR_BOOTSTRAP_SOURCE":
            verified, _ = verify_bootstrap_source(root)
            return verified
        if source_type in {
            "DETECTOR_CANONICAL_SOURCE",
            "AUXILIARY_CLASSIFICATION_SOURCE",
        }:
            verified, _ = verify_roboflow_source(root)
            return verified
    except (DetectorCorpusError, SampleDataAcquisitionError) as exc:
        raise DetectorCorpusError("additional source integrity verification failed") from exc
    raise DetectorCorpusError("additional source manifest type is unsupported")


def _safe_source_image(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise DetectorCorpusError("additional source image is missing or unsafe")
    return path


def _safe_target(root: Path, corpus_id: str) -> Path:
    target = (root / corpus_id).resolve()
    if target == root or not target.is_relative_to(root):
        raise DetectorCorpusError("plate corpus target escapes output root")
    return target


def _safe_child(root: Path, relative: str) -> Path:
    path_value = PurePosixPath(relative)
    if path_value.is_absolute() or not path_value.parts or ".." in path_value.parts:
        raise DetectorCorpusError("plate corpus path is unsafe")
    path = root.joinpath(*path_value.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorCorpusError("plate corpus path escapes root")
    return path


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
        raise DetectorCorpusError("refusing to remove an unsafe corpus path")
    if resolved.exists():
        shutil.rmtree(resolved)


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
    else:
        rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (rendered + "\n").encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> PlateCorpusBuildResult:
    return PlateCorpusBuildResult(
        corpus_id=str(manifest["corpusId"]),
        directory=directory,
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        duplicate_images_merged=int(manifest.get("statistics", {}).get("duplicateImagesMerged", 0)),
        reused=reused,
    )
