"""Build an immutable, review-only plate source from video extraction output.

The video extractor produces model suggestions in ``annotations.auto.jsonl``.
This adapter packages those suggestions into the same evidence-backed queue used
by the detector review UI, while keeping the source explicitly ineligible for
training promotion until its rights status is resolved outside this workflow.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from pydantic import ValidationError

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.domain import DetectorAnnotation, DetectorSample

VIDEO_REVIEW_SOURCE_TYPE = "VIDEO_DETECTOR_REVIEW_SOURCE"
VIDEO_REVIEW_REASON = "VIDEO_MODEL_SUGGESTION_REQUIRES_HUMAN_REVIEW"

_EXTRACTION_TYPE = "VIDEO_DETECTOR_SAMPLE_EXTRACTION"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})


@dataclass(frozen=True, slots=True)
class VideoPlateReviewSourceResult:
    source_id: str
    directory: Path
    manifest_sha256: str
    source_record_count: int
    review_queue_count: int
    suggestion_count: int
    exact_duplicate_images_merged: int
    reused: bool = False


@dataclass(slots=True)
class _ReviewImage:
    sha256: str
    source_path: Path
    suffix: str
    width: int
    height: int
    samples: list[DetectorSample] = field(default_factory=list)
    suggestions: dict[str, dict[str, Any]] = field(default_factory=dict)


class VideoPlateReviewSourceBuilder:
    """Convert a completed video extraction into a review-only UI source."""

    def __init__(
        self,
        *,
        extraction_directory: Path,
        output_directory: Path,
        source_id: str,
        owner_namespace: str,
        founder_id: str,
        clock: Any = lambda: datetime.now(UTC),
    ) -> None:
        if not _IDENTIFIER.fullmatch(source_id):
            raise ValueError("video review source id is not path-safe")
        self._extraction = extraction_directory.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._source_id = source_id
        self._owner_namespace = _required_text(owner_namespace, "owner namespace")
        self._founder_id = _required_text(founder_id, "founder id")
        self._clock = clock

    def build(self) -> VideoPlateReviewSourceResult:
        extraction_manifest_path = self._extraction / "manifest.json"
        extraction_manifest_raw, extraction_manifest = _load_extraction_manifest(
            extraction_manifest_path,
            owner_namespace=self._owner_namespace,
            founder_id=self._founder_id,
        )
        extraction_manifest_sha256 = _sha256(extraction_manifest_raw)
        if self._target.exists():
            manifest, digest = verify_video_plate_review_source(self._target)
            source_extraction = manifest.get("sourceExtraction", {})
            if (
                manifest.get("sourceId") != self._source_id
                or not isinstance(source_extraction, dict)
                or source_extraction.get("manifestSha256") != extraction_manifest_sha256
            ):
                raise DetectorDatasetError(
                    "existing video review source does not match this extraction"
                )
            return _result(self._target, manifest, digest, reused=True)
        if not self._extraction.is_dir() or self._extraction.is_symlink():
            raise DetectorDatasetError(
                f"video extraction directory is missing or unsafe: {self._extraction}"
            )
        if self._target == self._extraction or self._target.is_relative_to(self._extraction):
            raise DetectorDatasetError("video review source output cannot be inside extraction")

        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise DetectorDatasetError("video review source clock must be timezone-aware")
        now = now.astimezone(UTC)
        images, source_record_count = _load_review_images(
            self._extraction,
            extraction_manifest,
        )
        if not images:
            raise DetectorDatasetError("video extraction contains no plate review samples")

        self._target.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._target.parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary,
                extraction_manifest,
                extraction_manifest_raw,
                extraction_manifest_sha256,
                images,
                source_record_count,
                now,
            )
            manifest_raw = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_raw)
            if self._target.exists():
                raise DetectorDatasetError("video review source target already exists")
            temporary.replace(self._target)
            verified, digest = verify_video_plate_review_source(self._target)
            return _result(self._target, verified, digest)
        except DetectorDatasetError:
            _remove_temporary(temporary, self._target.parent)
            raise
        except Exception as exc:
            _remove_temporary(temporary, self._target.parent)
            raise DetectorDatasetError("cannot build video plate review source") from exc

    def _materialize(
        self,
        temporary: Path,
        extraction_manifest: dict[str, Any],
        extraction_manifest_raw: bytes,
        extraction_manifest_sha256: str,
        images: dict[str, _ReviewImage],
        source_record_count: int,
        now: datetime,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        review_lines: list[bytes] = []
        provenance_lines: list[bytes] = []
        duplicate_lines: list[bytes] = []
        lighting_counts: Counter[str] = Counter()
        suggestion_count = 0

        for digest, review_image in sorted(images.items()):
            relative = PurePosixPath(
                "review",
                "images",
                digest[:2],
                f"{digest}{review_image.suffix}",
            )
            destination = temporary.joinpath(*relative.parts)
            _copy_verified(review_image.source_path, destination, digest)
            files.append(_file_entry(destination, temporary))

            samples = sorted(review_image.samples, key=lambda item: item.sample_id)
            suggestions = sorted(
                review_image.suggestions.values(),
                key=lambda item: _suggestion_sort_key(item),
            )
            if not 1 <= len(suggestions) <= 16:
                raise DetectorDatasetError(
                    "video review image suggestion count exceeds UI contract"
                )
            suggestion_count += len(suggestions)
            for sample in samples:
                lighting_counts[str(sample.attributes.get("lighting", "UNKNOWN"))] += 1
            review_id = f"review-{digest[:24]}"
            review_lines.append(
                _json_bytes(
                    {
                        "schemaVersion": 1,
                        "reviewId": review_id,
                        "imagePath": str(relative),
                        "sourceImageSha256": digest,
                        "sourceFilenameSha256": _sha256(
                            samples[0].image_path.encode("utf-8")
                        ),
                        "reason": VIDEO_REVIEW_REASON,
                        "status": "PENDING_REVIEW",
                        "suggestions": suggestions,
                    },
                    pretty=False,
                )
            )
            provenance_records = [_sample_provenance(sample) for sample in samples]
            provenance_lines.append(
                _json_bytes(
                    {
                        "schemaVersion": 1,
                        "reviewId": review_id,
                        "sourceImageSha256": digest,
                        "records": provenance_records,
                    },
                    pretty=False,
                )
            )
            if len(samples) > 1:
                duplicate_lines.append(
                    _json_bytes(
                        {
                            "sourceImageSha256": digest,
                            "keptSampleId": samples[0].sample_id,
                            "mergedSampleIds": [sample.sample_id for sample in samples[1:]],
                            "duplicateCount": len(samples) - 1,
                        },
                        pretty=False,
                    )
                )

        evidence: tuple[tuple[str, bytes], ...] = (
            ("annotations.jsonl", b""),
            ("REVIEW_QUEUE.jsonl", b"".join(review_lines)),
            ("PROVENANCE.jsonl", b"".join(provenance_lines)),
            ("DUPLICATES.jsonl", b"".join(duplicate_lines)),
            ("EXTRACTION_MANIFEST.json", extraction_manifest_raw),
            (
                "SOURCE_CARD.md",
                _source_card(
                    source_id=self._source_id,
                    owner_namespace=self._owner_namespace,
                    founder_id=self._founder_id,
                    source_record_count=source_record_count,
                    review_queue_count=len(images),
                    suggestion_count=suggestion_count,
                ).encode("utf-8"),
            ),
        )
        for name, data in evidence:
            path = temporary / name
            _write_new(path, data)
            files.append(_file_entry(path, temporary))

        quality_audit = self._extraction / "plate" / "QUALITY_AUDIT.md"
        if quality_audit.is_file() and not quality_audit.is_symlink():
            destination = temporary / "QUALITY_AUDIT.md"
            _copy_verified(quality_audit, destination, _sha256_file(quality_audit))
            files.append(_file_entry(destination, temporary))

        video_sources = extraction_manifest["sources"]
        return {
            "schemaVersion": 1,
            "type": VIDEO_REVIEW_SOURCE_TYPE,
            "role": "plate",
            "sourceId": self._source_id,
            "ownerNamespace": self._owner_namespace,
            "founderId": self._founder_id,
            "createdAt": _timestamp(now),
            "collectionMethod": "EXTERNAL_VIDEO_EXTRACTION",
            "rightsAssertion": "UNRESOLVED_REQUIRES_REVIEW",
            "licenseStatus": "REVIEW_REQUIRED",
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "promotionEligible": False,
            "annotationPolicy": "MODEL_SUGGESTIONS_REQUIRE_HUMAN_REVIEW",
            "sampleCount": 0,
            "annotationCount": 0,
            "negativeSampleCount": 0,
            "reviewQueueCount": len(images),
            "suggestionCount": suggestion_count,
            "statistics": {
                "sourceRecordCount": source_record_count,
                "uniqueReviewImages": len(images),
                "exactDuplicateImagesMerged": source_record_count - len(images),
                "suggestionCount": suggestion_count,
                "sourceVideoCount": len(video_sources),
                "lightingRecordCounts": dict(sorted(lighting_counts.items())),
            },
            "sourceExtraction": {
                "type": _EXTRACTION_TYPE,
                "manifestPath": "EXTRACTION_MANIFEST.json",
                "manifestSha256": extraction_manifest_sha256,
                "createdAt": extraction_manifest["createdAt"],
                "sourceDirectoryName": extraction_manifest["sourceDirectoryName"],
                "videoSha256": sorted(str(item["sha256"]) for item in video_sources),
            },
            "models": extraction_manifest.get("models", {}),
            "files": sorted(files, key=lambda item: item["path"]),
        }


def verify_video_plate_review_source(directory: Path) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "source-manifest.json"
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("video review source manifest is invalid") from exc
    source_id = manifest.get("sourceId") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != VIDEO_REVIEW_SOURCE_TYPE
        or manifest.get("role") != "plate"
        or not isinstance(source_id, str)
        or not _IDENTIFIER.fullmatch(source_id)
        or root.name != source_id
        or manifest.get("licenseStatus") != "REVIEW_REQUIRED"
        or manifest.get("acceptanceEligible") is not False
        or manifest.get("releaseEligible") is not False
        or manifest.get("distributionEligible") is not False
        or manifest.get("promotionEligible") is not False
        or not isinstance(manifest.get("files"), list)
    ):
        raise DetectorDatasetError("video review source manifest contract is invalid")

    files: dict[str, dict[str, Any]] = {}
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DetectorDatasetError("video review source file entry is invalid")
        relative = str(entry["path"])
        path = _safe_child(root, relative)
        if relative in files or not path.is_file() or path.is_symlink():
            raise DetectorDatasetError("video review source file is missing or duplicated")
        if path.stat().st_size != int(entry.get("size", -1)):
            raise DetectorDatasetError("video review source file size verification failed")
        if _sha256_file(path) != entry.get("sha256"):
            raise DetectorDatasetError("video review source checksum verification failed")
        files[relative] = entry

    required = {
        "annotations.jsonl",
        "DUPLICATES.jsonl",
        "EXTRACTION_MANIFEST.json",
        "PROVENANCE.jsonl",
        "REVIEW_QUEUE.jsonl",
        "SOURCE_CARD.md",
    }
    if not required <= set(files):
        raise DetectorDatasetError("video review source evidence files are incomplete")
    if (root / "annotations.jsonl").read_bytes():
        raise DetectorDatasetError("review-only video source cannot contain training samples")

    queue_count = 0
    suggestion_count = 0
    review_ids: set[str] = set()
    image_hashes: set[str] = set()
    for line in (root / "REVIEW_QUEUE.jsonl").read_bytes().splitlines():
        if not line:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DetectorDatasetError("video review queue record is invalid") from exc
        image_sha = record.get("sourceImageSha256") if isinstance(record, dict) else None
        review_id = record.get("reviewId") if isinstance(record, dict) else None
        image_path = record.get("imagePath") if isinstance(record, dict) else None
        suggestions = record.get("suggestions") if isinstance(record, dict) else None
        if (
            record.get("schemaVersion") != 1
            or not isinstance(image_sha, str)
            or not _SHA256.fullmatch(image_sha)
            or review_id != f"review-{image_sha[:24]}"
            or review_id in review_ids
            or image_sha in image_hashes
            or not isinstance(image_path, str)
            or record.get("reason") != VIDEO_REVIEW_REASON
            or record.get("status") != "PENDING_REVIEW"
            or not isinstance(suggestions, list)
            or not 1 <= len(suggestions) <= 16
        ):
            raise DetectorDatasetError("video review queue contract is invalid")
        image_entry = files.get(image_path)
        if not isinstance(image_entry, dict) or image_entry.get("sha256") != image_sha:
            raise DetectorDatasetError("video review queue image binding is invalid")
        image = cv2.imread(str(_safe_child(root, image_path)))
        if image is None or image.size == 0:
            raise DetectorDatasetError("video review queue image cannot be decoded")
        height, width = image.shape[:2]
        for suggestion in suggestions:
            try:
                annotation = DetectorAnnotation.model_validate(suggestion)
            except ValidationError as exc:
                raise DetectorDatasetError("video review suggestion is invalid") from exc
            if annotation.class_name != "license_plate":
                raise DetectorDatasetError("video review suggestion class is invalid")
            box = annotation.bbox
            if box.x + box.width > width or box.y + box.height > height:
                raise DetectorDatasetError("video review suggestion is outside image bounds")
        review_ids.add(str(review_id))
        image_hashes.add(image_sha)
        queue_count += 1
        suggestion_count += len(suggestions)

    source_extraction = manifest.get("sourceExtraction")
    extraction_entry = files["EXTRACTION_MANIFEST.json"]
    if (
        not isinstance(source_extraction, dict)
        or source_extraction.get("manifestPath") != "EXTRACTION_MANIFEST.json"
        or source_extraction.get("manifestSha256") != extraction_entry.get("sha256")
    ):
        raise DetectorDatasetError("video review extraction evidence is invalid")
    if (
        manifest.get("sampleCount") != 0
        or manifest.get("annotationCount") != 0
        or manifest.get("negativeSampleCount") != 0
        or manifest.get("reviewQueueCount") != queue_count
        or manifest.get("suggestionCount") != suggestion_count
    ):
        raise DetectorDatasetError("video review source statistics do not match evidence")
    return manifest, _sha256(manifest_raw)


def _load_extraction_manifest(
    path: Path,
    *,
    owner_namespace: str,
    founder_id: str,
) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("video extraction manifest is invalid") from exc
    sources = manifest.get("sources") if isinstance(manifest, dict) else None
    statistics = manifest.get("statistics") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != _EXTRACTION_TYPE
        or manifest.get("status") != "COMPLETE"
        or manifest.get("ownerNamespace") != owner_namespace
        or manifest.get("founderId") != founder_id
        or manifest.get("licenseReviewStatus") != "REVIEW_REQUIRED"
        or manifest.get("acceptanceEligible") is not False
        or manifest.get("releaseEligible") is not False
        or manifest.get("distributionEligible") is not False
        or not isinstance(manifest.get("createdAt"), str)
        or not isinstance(manifest.get("sourceDirectoryName"), str)
        or not isinstance(sources, list)
        or not sources
        or not isinstance(statistics, dict)
        or not isinstance(statistics.get("plateTrainingImages"), int)
    ):
        raise DetectorDatasetError("video extraction manifest contract is invalid")
    for source in sources:
        if (
            not isinstance(source, dict)
            or source.get("status") != "PROCESSED"
            or not isinstance(source.get("sha256"), str)
            or not _SHA256.fullmatch(str(source["sha256"]))
            or source.get("licenseReviewStatus") != "REVIEW_REQUIRED"
            or source.get("releaseEligible") is not False
            or source.get("distributionEligible") is not False
        ):
            raise DetectorDatasetError("video extraction source evidence is invalid")
    return raw, manifest


def _load_review_images(
    extraction_root: Path,
    manifest: dict[str, Any],
) -> tuple[dict[str, _ReviewImage], int]:
    plate_root = extraction_root / "plate"
    annotations_path = plate_root / "annotations.auto.jsonl"
    if not annotations_path.is_file() or annotations_path.is_symlink():
        raise DetectorDatasetError("video plate suggestions are missing or unsafe")
    video_hashes = {str(source["sha256"]) for source in manifest["sources"]}
    images: dict[str, _ReviewImage] = {}
    source_record_count = 0
    for line in annotations_path.read_bytes().splitlines():
        if not line:
            continue
        try:
            sample = DetectorSample.model_validate_json(line)
        except ValidationError as exc:
            raise DetectorDatasetError("video plate suggestion record is invalid") from exc
        if (
            sample.attributes.get("annotationSource") != "MODEL_SUGGESTION"
            or sample.attributes.get("reviewStatus") != "PENDING_REVIEW"
            or sample.attributes.get("licenseReviewStatus") != "REVIEW_REQUIRED"
            or sample.attributes.get("acceptanceEligible") is not False
            or sample.attributes.get("releaseEligible") is not False
            or sample.attributes.get("sourceVideoSha256") not in video_hashes
            or not 1 <= len(sample.annotations) <= 16
        ):
            raise DetectorDatasetError("video plate suggestion eligibility is invalid")
        source_path = _safe_child(plate_root, sample.image_path)
        if (
            PurePosixPath(sample.image_path).parts[0] != "images"
            or not source_path.is_file()
            or source_path.is_symlink()
            or source_path.suffix.lower() not in _IMAGE_SUFFIXES
        ):
            raise DetectorDatasetError("video plate suggestion image is invalid")
        data = source_path.read_bytes()
        digest = _sha256(data)
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise DetectorDatasetError("video plate suggestion image cannot be decoded")
        height, width = image.shape[:2]
        review_image = images.get(digest)
        if review_image is None:
            review_image = _ReviewImage(
                sha256=digest,
                source_path=source_path,
                suffix=_canonical_suffix(source_path.suffix),
                width=width,
                height=height,
            )
            images[digest] = review_image
        elif (review_image.width, review_image.height) != (width, height):
            raise DetectorDatasetError("duplicate video review image dimensions conflict")
        review_image.samples.append(sample)
        for annotation in sample.annotations:
            if annotation.class_name != "license_plate":
                raise DetectorDatasetError("video plate suggestion class is invalid")
            box = annotation.bbox
            if box.x + box.width > width or box.y + box.height > height:
                raise DetectorDatasetError("video plate suggestion is outside image bounds")
            suggestion = _suggestion_json(annotation, sample.sample_id)
            signature = _suggestion_signature(suggestion)
            current = review_image.suggestions.get(signature)
            if current is None or _confidence(suggestion) > _confidence(current):
                review_image.suggestions[signature] = suggestion
        source_record_count += 1
    expected = manifest["statistics"]["plateTrainingImages"]
    if source_record_count != expected:
        raise DetectorDatasetError("video plate suggestion count does not match manifest")
    return images, source_record_count


def _suggestion_json(annotation: DetectorAnnotation, sample_id: str) -> dict[str, Any]:
    attributes = {
        key: value
        for key, value in annotation.attributes.items()
        if key != "cropPath"
        and (isinstance(value, (str, bool, int, float)) or value is None)
    }
    attributes["sourceSampleId"] = sample_id
    return {
        "className": "license_plate",
        "bbox": annotation.bbox.model_dump(mode="json"),
        "attributes": attributes,
    }


def _suggestion_signature(suggestion: dict[str, Any]) -> str:
    bbox = suggestion["bbox"]
    return ":".join(
        (
            str(suggestion["className"]),
            *(f"{float(bbox[field]):.6f}" for field in ("x", "y", "width", "height")),
        )
    )


def _suggestion_sort_key(suggestion: dict[str, Any]) -> tuple[float, float, float, float]:
    bbox = suggestion["bbox"]
    return (
        float(bbox["y"]),
        float(bbox["x"]),
        float(bbox["height"]),
        float(bbox["width"]),
    )


def _confidence(suggestion: dict[str, Any]) -> float:
    value = suggestion.get("attributes", {}).get("confidence", 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _sample_provenance(sample: DetectorSample) -> dict[str, Any]:
    attributes = sample.attributes
    return {
        "sampleId": sample.sample_id,
        "groupId": sample.group_id,
        "cameraId": sample.camera_id,
        "capturedAt": _timestamp(sample.captured_at),
        "sourceImagePathSha256": _sha256(sample.image_path.encode("utf-8")),
        "sourceVideo": attributes.get("sourceVideo"),
        "sourceVideoSha256": attributes.get("sourceVideoSha256"),
        "sourceFrameIndex": attributes.get("sourceFrameIndex"),
        "sourceOffsetSeconds": attributes.get("sourceOffsetSeconds"),
    }


def _source_card(
    *,
    source_id: str,
    owner_namespace: str,
    founder_id: str,
    source_record_count: int,
    review_queue_count: int,
    suggestion_count: int,
) -> str:
    return f"""# {source_id}

Immutable review-only Vietnam plate detector source built from local video
extraction evidence.

- Owner namespace: `{owner_namespace}`
- Founder/steward: `{founder_id}`
- Source records: {source_record_count}
- Unique review images: {review_queue_count}
- Model bbox suggestions: {suggestion_count}
- License/rights status: `REVIEW_REQUIRED`
- Promotion eligible: `false`
- Release eligible: `false`
- Distribution eligible: `false`

The model outputs in `REVIEW_QUEUE.jsonl` are suggestions, not ground truth.
Operators may review and correct bounding boxes in the Dataset Review UI. This
source cannot be promoted into a production training source until the source
rights are resolved and a separate eligible source is built with that evidence.
"""


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> VideoPlateReviewSourceResult:
    statistics = manifest["statistics"]
    return VideoPlateReviewSourceResult(
        source_id=str(manifest["sourceId"]),
        directory=directory,
        manifest_sha256=digest,
        source_record_count=int(statistics["sourceRecordCount"]),
        review_queue_count=int(manifest["reviewQueueCount"]),
        suggestion_count=int(manifest["suggestionCount"]),
        exact_duplicate_images_merged=int(statistics["exactDuplicateImagesMerged"]),
        reused=reused,
    )


def _file_entry(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _copy_verified(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with source.open("rb") as input_stream, destination.open("xb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
    except FileExistsError as exc:
        raise DetectorDatasetError("video review source image path collision") from exc
    if _sha256_file(destination) != expected_sha256:
        raise DetectorDatasetError("video review source copy checksum mismatch")


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _remove_temporary(path: Path, parent: Path) -> None:
    resolved = path.resolve()
    if (
        resolved.parent != parent.resolve()
        or not resolved.name.startswith(".")
        or ".tmp-" not in resolved.name
    ):
        raise DetectorDatasetError("refusing to remove unsafe video review directory")
    if resolved.exists():
        shutil.rmtree(resolved)


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("video review source path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("video review source path escapes its root")
    return path


def _canonical_suffix(value: str) -> str:
    return ".jpg" if value.lower() == ".jpeg" else value.lower()


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
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 128 or any(char in normalized for char in "/\\\0"):
        raise ValueError(f"{label} is invalid")
    return normalized


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
