"""Immutable first-party detector source ingestion with exact label recovery.

Production samples are admitted only when their exact image SHA-256 has an
existing canonical annotation. Model suggestions and unlabeled images remain in
an explicit review queue and never leak into ``annotations.jsonl``.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.corpus import verify_plate_corpus
from vehicle_intelligence.training.domain import (
    DetectorAnnotation,
    DetectorSample,
)

_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
_SOURCE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SOURCE_TYPE = "FIRST_PARTY_DETECTOR_SOURCE"
_LICENSE_STATUS = "PROPRIETARY_FIRST_PARTY_USER_CONFIRMED"
_ATTRIBUTION_FIELDS = (
    "sample_id",
    "source_dataset",
    "source_revision",
    "license",
    "author",
    "landing_url",
)


@dataclass(frozen=True, slots=True)
class FirstPartyPlateSourceResult:
    source_id: str
    directory: Path
    manifest_sha256: str
    sample_count: int
    annotation_count: int
    negative_sample_count: int
    review_queue_count: int
    exact_duplicate_files_excluded: int
    unsupported_file_count: int
    reused: bool = False


@dataclass(frozen=True, slots=True)
class _LabelReference:
    records: dict[str, DetectorSample]
    evidence_type: str
    evidence_id: str
    evidence_sha256: str


@dataclass(frozen=True, slots=True)
class _Inventory:
    canonical: dict[str, Path]
    duplicates: dict[str, tuple[Path, ...]]
    unsupported: tuple[Path, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class _AutoReference:
    records: dict[str, DetectorSample]
    conflicts: frozenset[str]


class FirstPartyPlateSourceBuilder:
    """Build a founder-namespaced source from user-confirmed first-party images."""

    def __init__(
        self,
        *,
        input_directory: Path,
        output_directory: Path,
        label_reference_directory: Path,
        source_id: str,
        owner_namespace: str,
        founder_id: str,
        auto_reference_directory: Path | None = None,
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        if not _SOURCE_ID.fullmatch(source_id):
            raise ValueError("first-party source id is not path-safe")
        self._input = input_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._label_reference = label_reference_directory.expanduser().resolve()
        self._auto_reference = (
            auto_reference_directory.expanduser().resolve()
            if auto_reference_directory is not None
            else None
        )
        self._source_id = source_id
        self._owner_namespace = _identifier(owner_namespace, "owner namespace")
        self._founder_id = _identifier(founder_id, "founder id")
        self._clock = clock

    def build(self) -> FirstPartyPlateSourceResult:
        if self._target.exists():
            manifest, digest = verify_first_party_detector_source(self._target)
            return _result(self._target, manifest, digest, reused=True)
        if not self._input.is_dir():
            raise DetectorDatasetError(
                f"first-party plate image directory is missing: {self._input}"
            )
        if self._target == self._input or self._target.is_relative_to(self._input):
            raise DetectorDatasetError("first-party source output cannot be inside its input")

        labels = _load_label_reference(self._label_reference)
        auto_labels = _load_auto_reference(self._auto_reference)
        inventory = _inventory(self._input)
        if not inventory.canonical:
            raise DetectorDatasetError("first-party plate source contains no decodable images")

        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise DetectorDatasetError("first-party source clock must be timezone-aware")
        now = now.astimezone(UTC)
        self._target.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._target.parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary,
                labels,
                auto_labels,
                inventory,
                now,
            )
            manifest_bytes = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_bytes)
            if self._target.exists():
                raise DetectorDatasetError("first-party source target already exists")
            temporary.replace(self._target)
            return _result(
                self._target,
                manifest,
                _sha256(manifest_bytes),
            )
        except DetectorDatasetError:
            _remove_tree(temporary, self._target.parent)
            raise
        except Exception as exc:
            _remove_tree(temporary, self._target.parent)
            raise DetectorDatasetError("cannot build first-party plate source") from exc

    def _materialize(
        self,
        temporary: Path,
        labels: _LabelReference,
        auto_labels: _AutoReference,
        inventory: _Inventory,
        now: datetime,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        annotation_records: list[bytes] = []
        review_records: list[bytes] = []
        attribution_rows: list[dict[str, str]] = []
        annotation_count = 0
        negative_sample_count = 0
        recovered_count = 0
        auto_review_count = 0
        auto_conflict_review_count = 0
        unlabeled_review_count = 0

        for digest, source_path in sorted(inventory.canonical.items()):
            data = source_path.read_bytes()
            if _sha256(data) != digest:
                raise DetectorDatasetError("first-party source changed during ingestion")
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                raise DetectorDatasetError(f"cannot decode first-party image: {source_path.name}")
            suffix = _canonical_suffix(source_path.suffix)
            reference = labels.records.get(digest)
            if reference is not None:
                relative = PurePosixPath("images", digest[:2], f"{digest}{suffix}")
                destination = temporary.joinpath(*relative.parts)
                _copy_new(source_path, destination)
                files.append(_known_file_entry(relative, digest, len(data)))
                sample = self._production_sample(
                    digest=digest,
                    relative=relative,
                    source_path=source_path,
                    image=image,
                    reference=reference,
                    label_reference=labels,
                )
                annotation_records.append(
                    _json_bytes(sample.model_dump(mode="json", by_alias=True), pretty=False)
                )
                annotation_count += len(sample.annotations)
                negative_sample_count += not bool(sample.annotations)
                recovered_count += 1
                attribution_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "source_dataset": self._source_id,
                        "source_revision": "v1",
                        "license": "PROPRIETARY-FIRST-PARTY",
                        "author": self._founder_id,
                        "landing_url": "",
                    }
                )
                continue

            relative = PurePosixPath("review", "images", digest[:2], f"{digest}{suffix}")
            destination = temporary.joinpath(*relative.parts)
            _copy_new(source_path, destination)
            files.append(_known_file_entry(relative, digest, len(data)))
            suggestion = auto_labels.records.get(digest)
            if digest in auto_labels.conflicts:
                reason = "AUTO_LABEL_CONFLICT_REQUIRES_HUMAN_REVIEW"
            elif suggestion is not None:
                reason = "MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW"
            else:
                reason = "MISSING_VERIFIED_ANNOTATION"
            auto_review_count += suggestion is not None
            auto_conflict_review_count += digest in auto_labels.conflicts
            unlabeled_review_count += suggestion is None and digest not in auto_labels.conflicts
            review_records.append(
                _json_bytes(
                    {
                        "schemaVersion": 1,
                        "reviewId": f"review-{digest[:24]}",
                        "imagePath": str(relative),
                        "sourceImageSha256": digest,
                        "sourceFilenameSha256": _sha256(source_path.name.encode()),
                        "reason": reason,
                        "status": "PENDING_REVIEW",
                        "suggestions": (
                            [
                                annotation.model_dump(mode="json", by_alias=True)
                                for annotation in suggestion.annotations
                            ]
                            if suggestion is not None
                            else []
                        ),
                    },
                    pretty=False,
                )
            )

        annotations_path = temporary / "annotations.jsonl"
        _write_new(annotations_path, b"".join(annotation_records))
        files.append(_file_entry(annotations_path, temporary))
        review_path = temporary / "REVIEW_QUEUE.jsonl"
        _write_new(review_path, b"".join(review_records))
        files.append(_file_entry(review_path, temporary))
        duplicates_path = temporary / "DUPLICATES.jsonl"
        _write_new(duplicates_path, _duplicates(inventory, self._input))
        files.append(_file_entry(duplicates_path, temporary))
        rejects_path = temporary / "REJECTS.jsonl"
        _write_new(rejects_path, _unsupported(inventory, self._input))
        files.append(_file_entry(rejects_path, temporary))
        attribution_path = temporary / "ATTRIBUTION.csv"
        _write_new(attribution_path, _attribution(attribution_rows))
        files.append(_file_entry(attribution_path, temporary))
        card_path = temporary / "SOURCE_CARD.md"
        _write_new(
            card_path,
            _source_card(
                source_id=self._source_id,
                owner_namespace=self._owner_namespace,
                founder_id=self._founder_id,
                production_count=recovered_count,
                review_count=len(review_records),
            ).encode(),
        )
        files.append(_file_entry(card_path, temporary))

        return {
            "schemaVersion": 1,
            "type": _SOURCE_TYPE,
            "role": "plate",
            "sourceId": self._source_id,
            "ownerNamespace": self._owner_namespace,
            "founderId": self._founder_id,
            "createdAt": _timestamp(now),
            "collectionMethod": "FIRST_PARTY_USER_COLLECTED",
            "rightsAssertion": "USER_CONFIRMED_FIRST_PARTY_COLLECTION",
            "licenseStatus": _LICENSE_STATUS,
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": True,
            "releaseEligible": True,
            "distributionEligible": False,
            "annotationPolicy": "EXACT_SHA256_RECOVERED_CANONICAL_LABELS_ONLY",
            "sampleCount": recovered_count,
            "annotationCount": annotation_count,
            "negativeSampleCount": negative_sample_count,
            "reviewQueueCount": len(review_records),
            "statistics": {
                "inputImageFiles": sum(len(paths) for paths in inventory.duplicates.values()),
                "uniqueImages": len(inventory.canonical),
                "verifiedProductionImages": recovered_count,
                "autoLabeledPendingReview": auto_review_count,
                "autoLabelConflictsPendingReview": auto_conflict_review_count,
                "unlabeledPendingReview": unlabeled_review_count,
                "exactDuplicateFilesExcluded": sum(
                    len(paths) - 1 for paths in inventory.duplicates.values()
                ),
                "unsupportedFiles": len(inventory.unsupported),
            },
            "inputInventorySha256": inventory.digest,
            "labelReference": {
                "type": labels.evidence_type,
                "id": labels.evidence_id,
                "sha256": labels.evidence_sha256,
            },
            "files": sorted(files, key=lambda item: item["path"]),
        }

    def _production_sample(
        self,
        *,
        digest: str,
        relative: PurePosixPath,
        source_path: Path,
        image: np.ndarray,
        reference: DetectorSample,
        label_reference: _LabelReference,
    ) -> DetectorSample:
        height, width = image.shape[:2]
        _validate_reference_annotations(reference, width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        annotations = tuple(
            DetectorAnnotation(
                className="license_plate",
                bbox=annotation.bbox,
                polygon=annotation.polygon,
                attributes={
                    "annotationOrigin": "EXACT_SHA256_CANONICAL_REFERENCE",
                    "reviewStatus": "VERIFIED_EXISTING_LABEL",
                    "referenceAnnotationSha256": _sha256(
                        _json_bytes(
                            annotation.model_dump(mode="json", by_alias=True),
                            pretty=False,
                        )
                    ),
                },
            )
            for annotation in reference.annotations
        )
        group_digest = _sha256(reference.group_id.encode())[:32]
        return DetectorSample(
            sampleId=f"phins-first-party-plate-{digest[:24]}",
            imagePath=str(relative),
            groupId=f"phins-group:first-party-plate:{group_digest}",
            cameraId="first-party-collection",
            capturedAt=datetime.fromtimestamp(source_path.stat().st_mtime, tz=UTC),
            split=None,
            attributes={
                "ownerNamespace": self._owner_namespace,
                "founderId": self._founder_id,
                "sourceCollection": "FIRST_PARTY_USER_COLLECTED",
                "sourceOwner": self._founder_id,
                "sourceLicense": "PROPRIETARY-FIRST-PARTY",
                "sourceImageSha256": digest,
                "sourceFilenameSha256": _sha256(source_path.name.encode()),
                "capturedAtBasis": "SOURCE_FILE_MTIME",
                "actualCaptureTimeKnown": False,
                "annotationOrigin": "EXACT_SHA256_CANONICAL_REFERENCE",
                "annotationReviewStatus": "VERIFIED_EXISTING_LABEL",
                "labelReferenceType": label_reference.evidence_type,
                "labelReferenceId": label_reference.evidence_id,
                "acceptanceEligible": True,
                "releaseEligible": True,
                "distributionEligible": False,
                "negativeSample": not bool(annotations),
                "lighting": "NIGHT" if brightness < 70 else "DAY",
                "imageBrightness": round(brightness, 4),
                "imageContrast": round(float(np.std(gray)), 4),
                "imageSharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 4),
            },
            annotations=annotations,
        )


def verify_first_party_detector_source(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "source-manifest.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("first-party source manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != _SOURCE_TYPE
        or manifest.get("role") != "plate"
        or manifest.get("acceptanceEligible") is not True
        or manifest.get("releaseEligible") is not True
        or manifest.get("distributionEligible") is not False
        or manifest.get("licenseStatus") != _LICENSE_STATUS
        or not isinstance(manifest.get("files"), list)
    ):
        raise DetectorDatasetError("first-party source manifest contract is invalid")

    recorded: set[str] = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DetectorDatasetError("first-party source file entry is invalid")
        relative = entry["path"]
        path = _safe_child(root, relative)
        if relative in recorded or not path.is_file() or path.is_symlink():
            raise DetectorDatasetError("first-party source file is missing or duplicated")
        recorded.add(relative)
        if path.stat().st_size != int(entry.get("size", -1)):
            raise DetectorDatasetError("first-party source file size verification failed")
        if _sha256_file(path) != entry.get("sha256"):
            raise DetectorDatasetError("first-party source checksum verification failed")

    required = {
        "annotations.jsonl",
        "ATTRIBUTION.csv",
        "DUPLICATES.jsonl",
        "REJECTS.jsonl",
        "REVIEW_QUEUE.jsonl",
        "SOURCE_CARD.md",
    }
    if not required <= recorded:
        raise DetectorDatasetError("first-party source evidence files are incomplete")
    sample_count = 0
    annotation_count = 0
    negative_count = 0
    image_paths: set[str] = set()
    for line in (root / "annotations.jsonl").read_bytes().splitlines():
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorDatasetError("first-party source annotation is invalid") from exc
        if (
            sample.attributes.get("acceptanceEligible") is not True
            or sample.attributes.get("releaseEligible") is not True
            or sample.attributes.get("distributionEligible") is not False
        ):
            raise DetectorDatasetError("first-party sample eligibility is invalid")
        if sample.image_path not in recorded or sample.image_path in image_paths:
            raise DetectorDatasetError("first-party source image reference is invalid")
        image_paths.add(sample.image_path)
        image = cv2.imread(str(_safe_child(root, sample.image_path)))
        if image is None or image.size == 0:
            raise DetectorDatasetError("first-party source image cannot be decoded")
        height, width = image.shape[:2]
        _validate_reference_annotations(sample, width, height)
        sample_count += 1
        annotation_count += len(sample.annotations)
        negative_count += not bool(sample.annotations)

    review_count = sum(
        1 for line in (root / "REVIEW_QUEUE.jsonl").read_bytes().splitlines() if line
    )
    if (
        sample_count != int(manifest.get("sampleCount", -1))
        or annotation_count != int(manifest.get("annotationCount", -1))
        or negative_count != int(manifest.get("negativeSampleCount", -1))
        or review_count != int(manifest.get("reviewQueueCount", -1))
    ):
        raise DetectorDatasetError("first-party source statistics do not match manifest")
    return manifest, _sha256(raw)


def _load_label_reference(root: Path) -> _LabelReference:
    annotations_path = root / "annotations.jsonl"
    if not annotations_path.is_file():
        raise DetectorDatasetError("canonical plate label reference is missing")
    evidence_type = "CANONICAL_ANNOTATIONS"
    evidence_id = root.name
    evidence_sha256 = _sha256(annotations_path.read_bytes())
    path_digests: dict[str, str] = {}
    corpus_path = root / "corpus-manifest.json"
    if corpus_path.is_file():
        corpus_manifest, evidence_sha256 = verify_plate_corpus(root)
        evidence_type = "VERIFIED_PLATE_CORPUS"
        evidence_id = str(corpus_manifest["corpusId"])
        path_digests = {
            str(entry["path"]): str(entry["sha256"])
            for entry in corpus_manifest["files"]
            if isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and str(entry["path"]).startswith("images/")
        }

    records: dict[str, DetectorSample] = {}
    for line in annotations_path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorDatasetError("canonical plate label reference is invalid") from exc
        if any(annotation.class_name != "license_plate" for annotation in sample.annotations):
            raise DetectorDatasetError("plate label reference contains another class")
        image_path = _safe_child(root, sample.image_path)
        digest = path_digests.get(sample.image_path) or _sha256_file(image_path)
        existing = records.get(digest)
        if existing is not None and _annotation_signature(existing) != _annotation_signature(
            sample
        ):
            raise DetectorDatasetError("exact label reference image has conflicting annotations")
        records.setdefault(digest, sample)
    if not records:
        raise DetectorDatasetError("canonical plate label reference is empty")
    return _LabelReference(records, evidence_type, evidence_id, evidence_sha256)


def _load_auto_reference(root: Path | None) -> _AutoReference:
    if root is None:
        return _AutoReference({}, frozenset())
    annotations_path = root / "annotations.auto.jsonl"
    if not annotations_path.is_file():
        raise DetectorDatasetError("auto-label review reference is missing")
    records: dict[str, DetectorSample] = {}
    conflicts: set[str] = set()
    for line in annotations_path.read_bytes().splitlines():
        if not line.strip():
            continue
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorDatasetError("auto-label review reference is invalid") from exc
        image_path = _safe_child(root, sample.image_path)
        digest = _sha256_file(image_path)
        existing = records.get(digest)
        if existing is not None and _annotation_signature(existing) != _annotation_signature(
            sample
        ):
            records.pop(digest, None)
            conflicts.add(digest)
            continue
        if digest not in conflicts:
            records.setdefault(digest, sample)
    return _AutoReference(records, frozenset(conflicts))


def _inventory(root: Path) -> _Inventory:
    grouped: dict[str, list[Path]] = defaultdict(list)
    unsupported: list[Path] = []
    digest_lines: list[str] = []
    for path in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root).as_posix()
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            unsupported.append(path)
            digest_lines.append(f"UNSUPPORTED\t{relative}\t{_sha256_file(path)}")
            continue
        digest = _sha256_file(path)
        grouped[digest].append(path)
        digest_lines.append(f"IMAGE\t{relative}\t{digest}")
    canonical = {digest: sorted(paths)[0] for digest, paths in grouped.items()}
    duplicates = {digest: tuple(sorted(paths)) for digest, paths in grouped.items()}
    return _Inventory(
        canonical=canonical,
        duplicates=duplicates,
        unsupported=tuple(unsupported),
        digest=_sha256(("\n".join(digest_lines) + "\n").encode()),
    )


def _validate_reference_annotations(sample: DetectorSample, width: int, height: int) -> None:
    for annotation in sample.annotations:
        if annotation.class_name != "license_plate":
            raise DetectorDatasetError("production plate sample contains another class")
        bbox = annotation.bbox
        if bbox.x + bbox.width > width + 1e-6 or bbox.y + bbox.height > height + 1e-6:
            raise DetectorDatasetError("production plate bounding box exceeds image")
        if any(point.x > width + 1e-6 or point.y > height + 1e-6 for point in annotation.polygon):
            raise DetectorDatasetError("production plate polygon exceeds image")


def _annotation_signature(sample: DetectorSample) -> str:
    value = [annotation.model_dump(mode="json", by_alias=True) for annotation in sample.annotations]
    return _sha256(_json_bytes(value, pretty=False))


def _duplicates(inventory: _Inventory, root: Path) -> bytes:
    records = []
    for digest, paths in sorted(inventory.duplicates.items()):
        if len(paths) <= 1:
            continue
        records.append(
            _json_bytes(
                {
                    "sha256": digest,
                    "keptFilenameSha256": _sha256(paths[0].name.encode()),
                    "duplicateFilenameSha256": [_sha256(path.name.encode()) for path in paths[1:]],
                    "duplicateCount": len(paths) - 1,
                },
                pretty=False,
            )
        )
    return b"".join(records)


def _unsupported(inventory: _Inventory, root: Path) -> bytes:
    return b"".join(
        _json_bytes(
            {
                "filenameSha256": _sha256(path.name.encode()),
                "reason": "UNSUPPORTED_NON_IMAGE_FILE",
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            },
            pretty=False,
        )
        for path in inventory.unsupported
    )


def _attribution(rows: list[dict[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=_ATTRIBUTION_FIELDS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(sorted(rows, key=lambda row: row["sample_id"]))
    return stream.getvalue().encode()


def _source_card(
    *,
    source_id: str,
    owner_namespace: str,
    founder_id: str,
    production_count: int,
    review_count: int,
) -> str:
    return f"""# {source_id}

Immutable first-party Vietnam plate detector source.

- Owner namespace: `{owner_namespace}`
- Founder/steward: `{founder_id}`
- Collection method: `FIRST_PARTY_USER_COLLECTED`
- License: `PROPRIETARY-FIRST-PARTY`
- Production samples: {production_count}
- Pending review: {review_count}
- Dataset redistribution: disabled

Only images with exact SHA-256 matches to canonical annotations are present in
`annotations.jsonl`. Model suggestions and unlabeled images are isolated in
`REVIEW_QUEUE.jsonl` and cannot enter detector training until reviewed.
"""


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> FirstPartyPlateSourceResult:
    statistics = manifest.get("statistics", {})
    return FirstPartyPlateSourceResult(
        source_id=str(manifest["sourceId"]),
        directory=directory,
        manifest_sha256=digest,
        sample_count=int(manifest["sampleCount"]),
        annotation_count=int(manifest["annotationCount"]),
        negative_sample_count=int(manifest["negativeSampleCount"]),
        review_queue_count=int(manifest["reviewQueueCount"]),
        exact_duplicate_files_excluded=int(statistics["exactDuplicateFilesExcluded"]),
        unsupported_file_count=int(statistics["unsupportedFiles"]),
        reused=reused,
    )


def _known_file_entry(relative: PurePosixPath, digest: str, size: int) -> dict[str, Any]:
    return {"path": str(relative), "sha256": digest, "size": size}


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    return {"path": relative, "sha256": _sha256_file(path), "size": path.stat().st_size}


def _copy_new(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
    except FileExistsError as exc:
        raise DetectorDatasetError("first-party canonical image path collision") from exc


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("first-party source path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("first-party source path escapes its root")
    return path


def _canonical_suffix(value: str) -> str:
    lowered = value.lower()
    return ".jpg" if lowered == ".jpeg" else lowered


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            indent=2 if pretty else None,
            sort_keys=True,
            ensure_ascii=False,
            separators=None if pretty else (",", ":"),
        )
        + "\n"
    ).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identifier(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128 or any(char in normalized for char in "/\\\0"):
        raise ValueError(f"{label} is invalid")
    return normalized


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _remove_tree(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if resolved.parent != parent.resolve() or not resolved.name.startswith("."):
        raise DetectorDatasetError("refusing to remove unsafe first-party temporary directory")
    if resolved.exists():
        shutil.rmtree(resolved)
