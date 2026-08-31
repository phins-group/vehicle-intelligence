"""Stage cleaned warehouse frames as an immutable plate-review source."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np

from vehicle_intelligence.exceptions import DetectorDatasetError
from vehicle_intelligence.training.warehouse_vehicle import (
    WarehouseArchiveScan,
    WarehouseImageCandidate,
    WarehouseVehicleImportOptions,
    clean_warehouse_image,
    deduplicate_warehouse_candidates,
    exact_warehouse_duplicate_records,
    scan_warehouse_archive,
)

WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE = "WAREHOUSE_PLATE_REVIEW_SOURCE"
WAREHOUSE_PLATE_REVIEW_REASON = "WAREHOUSE_IMAGE_REQUIRES_PLATE_REVIEW"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class WarehousePlateReviewSourceResult:
    source_id: str
    directory: Path
    manifest_sha256: str
    archive_sha256: str
    archive_image_count: int
    unique_raw_image_count: int
    review_queue_count: int
    exact_duplicate_files_excluded: int
    near_duplicate_images_excluded: int
    post_clean_duplicate_images_excluded: int
    rejected_unique_images: int
    reused: bool = False


class WarehousePlateReviewSourceBuilder:
    """Clean and deduplicate warehouse captures before human plate labeling."""

    def __init__(
        self,
        *,
        archive_path: Path,
        output_directory: Path,
        source_id: str,
        owner_namespace: str,
        founder_id: str,
        options: WarehouseVehicleImportOptions | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        if not _IDENTIFIER.fullmatch(source_id):
            raise ValueError("warehouse plate review source id is not path-safe")
        self._archive = archive_path.expanduser().resolve()
        self._target = output_directory.expanduser().resolve()
        self._source_id = source_id
        self._owner_namespace = _required_text(owner_namespace, "owner namespace")
        self._founder_id = _required_text(founder_id, "founder id")
        self._options = options or WarehouseVehicleImportOptions()
        self._clock = clock

    def build(self) -> WarehousePlateReviewSourceResult:
        if not self._archive.is_file() or self._archive.is_symlink():
            raise DetectorDatasetError(
                f"warehouse image archive is missing or unsafe: {self._archive}"
            )
        archive_sha256 = _sha256_file(self._archive)
        if self._target.exists():
            manifest, digest = verify_warehouse_plate_review_source(self._target)
            source_archive = manifest.get("sourceArchive", {})
            if (
                manifest.get("sourceId") != self._source_id
                or not isinstance(source_archive, dict)
                or source_archive.get("sha256") != archive_sha256
            ):
                raise DetectorDatasetError(
                    "existing warehouse plate review source does not match this archive"
                )
            return _result(self._target, manifest, digest, reused=True)

        now = self._clock()
        if not isinstance(now, datetime) or now.tzinfo is None:
            raise DetectorDatasetError("warehouse plate review clock must be timezone-aware")
        scan = scan_warehouse_archive(self._archive, self._options)
        selected, near_duplicates = deduplicate_warehouse_candidates(
            list(scan.candidates),
            self._options,
        )
        if not selected:
            raise DetectorDatasetError("warehouse archive has no unique reviewable images")

        parent = self._target.parent
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f".{self._target.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(parents=False, exist_ok=False)
            manifest = self._materialize(
                temporary=temporary,
                archive_sha256=archive_sha256,
                scan=scan,
                selected=selected,
                near_duplicates=near_duplicates,
                now=now.astimezone(UTC),
            )
            manifest_raw = _json_bytes(manifest, pretty=True)
            _write_new(temporary / "source-manifest.json", manifest_raw)
            if self._target.exists():
                raise DetectorDatasetError("warehouse plate review source target already exists")
            temporary.replace(self._target)
            verified, digest = verify_warehouse_plate_review_source(self._target)
            return _result(self._target, verified, digest)
        except DetectorDatasetError:
            _remove_temporary(temporary, parent)
            raise
        except Exception as exc:
            _remove_temporary(temporary, parent)
            raise DetectorDatasetError("cannot build warehouse plate review source") from exc

    def _materialize(
        self,
        *,
        temporary: Path,
        archive_sha256: str,
        scan: WarehouseArchiveScan,
        selected: list[WarehouseImageCandidate],
        near_duplicates: list[dict[str, Any]],
        now: datetime,
    ) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        queue_lines: list[bytes] = []
        provenance_lines: list[bytes] = []
        post_clean_duplicates: list[dict[str, Any]] = []
        cleaned_digests: dict[str, WarehouseImageCandidate] = {}
        by_name = {candidate.member.name: candidate for candidate in selected}

        try:
            with tarfile.open(self._archive, mode="r:gz") as archive:
                for tar_info in archive:
                    candidate = by_name.get(tar_info.name)
                    if candidate is None:
                        continue
                    stream = archive.extractfile(tar_info)
                    if stream is None:
                        raise DetectorDatasetError("selected warehouse review image cannot be read")
                    raw = stream.read(self._options.maximum_member_bytes + 1)
                    if _sha256(raw) != candidate.raw_sha256:
                        raise DetectorDatasetError(
                            "warehouse archive changed during plate-review staging"
                        )
                    image = cv2.imdecode(
                        np.frombuffer(raw, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if image is None or image.size == 0:
                        raise DetectorDatasetError(
                            "selected warehouse review image cannot be decoded"
                        )
                    cleaned = clean_warehouse_image(image, candidate.bbox)
                    ok, encoded = cv2.imencode(
                        ".jpg",
                        cleaned,
                        [cv2.IMWRITE_JPEG_QUALITY, self._options.jpeg_quality],
                    )
                    if not ok:
                        raise DetectorDatasetError("warehouse review image cannot be encoded")
                    image_bytes = encoded.tobytes()
                    image_sha256 = _sha256(image_bytes)
                    previous = cleaned_digests.get(image_sha256)
                    if previous is not None:
                        post_clean_duplicates.append(
                            {
                                "reason": "POST_CLEAN_EXACT_DUPLICATE",
                                "cleanedSha256": image_sha256,
                                "keptRawSha256": previous.raw_sha256,
                                "excludedRawSha256": candidate.raw_sha256,
                                "keptArchiveMember": previous.member.name,
                                "excludedArchiveMember": candidate.member.name,
                            }
                        )
                        continue
                    cleaned_digests[image_sha256] = candidate
                    relative = PurePosixPath(
                        "review",
                        "images",
                        image_sha256[:2],
                        f"{image_sha256}.jpg",
                    )
                    destination = temporary.joinpath(*relative.parts)
                    _write_new(destination, image_bytes)
                    files.append(_file_entry(destination, temporary))
                    review_id = f"review-{image_sha256[:24]}"
                    queue_lines.append(
                        _json_bytes(
                            {
                                "schemaVersion": 1,
                                "reviewId": review_id,
                                "imagePath": str(relative),
                                "sourceImageSha256": image_sha256,
                                "sourceFilenameSha256": _sha256(
                                    candidate.member.name.encode("utf-8")
                                ),
                                "reason": WAREHOUSE_PLATE_REVIEW_REASON,
                                "status": "PENDING_REVIEW",
                                "suggestions": [],
                            },
                            pretty=False,
                        )
                    )
                    provenance_lines.append(
                        _json_bytes(
                            _provenance_record(
                                review_id=review_id,
                                image_sha256=image_sha256,
                                archive_sha256=archive_sha256,
                                candidate=candidate,
                                scan=scan,
                            ),
                            pretty=False,
                        )
                    )
        except DetectorDatasetError:
            raise
        except (OSError, tarfile.TarError, cv2.error) as exc:
            raise DetectorDatasetError("cannot materialize warehouse plate review images") from exc
        if len(cleaned_digests) + len(post_clean_duplicates) != len(selected):
            raise DetectorDatasetError(
                "selected warehouse review images are missing from the archive"
            )

        exact_duplicates = exact_warehouse_duplicate_records(scan.names_by_digest)
        duplicate_records = [*exact_duplicates, *near_duplicates, *post_clean_duplicates]
        reject_records = [
            {
                "schemaVersion": 1,
                "reason": rejected.reason,
                "archiveMember": rejected.member.name,
                "sourceImageSha256": rejected.raw_sha256,
                "size": rejected.member.size,
            }
            for rejected in sorted(scan.rejects, key=lambda item: item.raw_sha256)
        ]
        evidence: tuple[tuple[str, bytes], ...] = (
            ("annotations.jsonl", b""),
            ("REVIEW_QUEUE.jsonl", b"".join(queue_lines)),
            ("PROVENANCE.jsonl", b"".join(provenance_lines)),
            ("DUPLICATES.jsonl", _json_lines(duplicate_records)),
            ("REJECTS.jsonl", _json_lines(reject_records)),
            (
                "SOURCE_CARD.md",
                _source_card(
                    source_id=self._source_id,
                    review_count=len(queue_lines),
                    exact_duplicates=len(exact_duplicates),
                    near_duplicates=len(near_duplicates),
                    post_clean_duplicates=len(post_clean_duplicates),
                    rejected=len(reject_records),
                ).encode("utf-8"),
            ),
        )
        for name, payload in evidence:
            path = temporary / name
            _write_new(path, payload)
            files.append(_file_entry(path, temporary))

        return {
            "schemaVersion": 1,
            "type": WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE,
            "role": "plate",
            "sourceId": self._source_id,
            "ownerNamespace": self._owner_namespace,
            "founderId": self._founder_id,
            "createdAt": _timestamp(now),
            "collectionMethod": "WAREHOUSE_CAMERA_ARCHIVE",
            "rightsAssertion": "UNRESOLVED_REQUIRES_REVIEW",
            "licenseStatus": "REVIEW_REQUIRED",
            "privacyClassification": "RESTRICTED_VEHICLE_IDENTIFIER",
            "acceptanceEligible": False,
            "releaseEligible": False,
            "distributionEligible": False,
            "promotionEligible": False,
            "annotationPolicy": "CLEANED_DEDUPLICATED_IMAGES_REQUIRE_HUMAN_PLATE_REVIEW",
            "sampleCount": 0,
            "annotationCount": 0,
            "negativeSampleCount": 0,
            "reviewQueueCount": len(queue_lines),
            "suggestionCount": 0,
            "sourceArchive": {
                "sha256": archive_sha256,
                "memberCount": scan.member_count,
                "imageCount": scan.image_count,
                "declaredImageBytes": scan.declared_bytes,
            },
            "deduplicationPolicy": {
                "exactSha256": True,
                "postCleanExactSha256": True,
                "perceptual": {
                    "maximumPhashDistance": self._options.maximum_phash_distance,
                    "maximumDhashDistance": self._options.maximum_dhash_distance,
                    "maximumThumbnailMae": self._options.maximum_thumbnail_mae,
                    "minimumThumbnailCorrelation": (self._options.minimum_thumbnail_correlation),
                    "minimumEdgeDice": self._options.minimum_edge_dice,
                    "sameCameraViewOnly": True,
                },
            },
            "statistics": {
                "uniqueRawImages": len(scan.names_by_digest),
                "recoverableImages": len(scan.candidates),
                "exactDuplicateFilesExcluded": len(exact_duplicates),
                "perceptualNearDuplicateImagesExcluded": len(near_duplicates),
                "postCleanExactDuplicateImagesExcluded": len(post_clean_duplicates),
                "rejectedUniqueImages": len(reject_records),
                "uniqueReviewImages": len(queue_lines),
            },
            "files": sorted(files, key=lambda item: item["path"]),
        }


def verify_warehouse_plate_review_source(
    directory: Path,
) -> tuple[dict[str, Any], str]:
    root = directory.expanduser().resolve()
    manifest_path = root / "source-manifest.json"
    try:
        manifest_raw = manifest_path.read_bytes()
        manifest = json.loads(manifest_raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DetectorDatasetError("warehouse plate review source manifest is invalid") from exc
    _validate_manifest_contract(root, manifest)
    files = _verify_files(root, manifest["files"])
    _verify_required_evidence(root, files)
    queue_count = _verify_queue(root, files)
    duplicate_records = _read_json_records(root / "DUPLICATES.jsonl")
    reject_count = len(_read_json_records(root / "REJECTS.jsonl"))
    _verify_statistics(manifest, queue_count, duplicate_records, reject_count)
    return manifest, _sha256(manifest_raw)


def _validate_manifest_contract(root: Path, manifest: object) -> None:
    source_id = manifest.get("sourceId") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schemaVersion") != 1
        or manifest.get("type") != WAREHOUSE_PLATE_REVIEW_SOURCE_TYPE
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
        raise DetectorDatasetError("warehouse plate review source manifest contract is invalid")


def _verify_required_evidence(root: Path, files: dict[str, dict[str, Any]]) -> None:
    required = {
        "annotations.jsonl",
        "DUPLICATES.jsonl",
        "PROVENANCE.jsonl",
        "REJECTS.jsonl",
        "REVIEW_QUEUE.jsonl",
        "SOURCE_CARD.md",
    }
    if not required <= set(files):
        raise DetectorDatasetError("warehouse plate review source evidence files are incomplete")
    if (root / "annotations.jsonl").read_bytes():
        raise DetectorDatasetError("review-only warehouse source cannot contain training samples")


def _verify_queue(root: Path, files: dict[str, dict[str, Any]]) -> int:
    queue = _read_jsonl(root / "REVIEW_QUEUE.jsonl", "reviewId")
    provenance = _read_jsonl(root / "PROVENANCE.jsonl", "reviewId")
    if set(queue) != set(provenance):
        raise DetectorDatasetError("warehouse review queue and provenance do not match")
    image_hashes: set[str] = set()
    for review_id, record in queue.items():
        image_sha256 = record.get("sourceImageSha256")
        image_path = record.get("imagePath")
        if (
            record.get("schemaVersion") != 1
            or not isinstance(image_sha256, str)
            or not _SHA256.fullmatch(image_sha256)
            or review_id != f"review-{image_sha256[:24]}"
            or image_sha256 in image_hashes
            or not isinstance(image_path, str)
            or not isinstance(record.get("sourceFilenameSha256"), str)
            or not _SHA256.fullmatch(str(record["sourceFilenameSha256"]))
            or record.get("reason") != WAREHOUSE_PLATE_REVIEW_REASON
            or record.get("status") != "PENDING_REVIEW"
            or record.get("suggestions") != []
        ):
            raise DetectorDatasetError("warehouse plate review queue contract is invalid")
        image_entry = files.get(image_path)
        if not isinstance(image_entry, dict) or image_entry.get("sha256") != image_sha256:
            raise DetectorDatasetError("warehouse plate review queue image binding is invalid")
        image = cv2.imread(str(_safe_child(root, image_path)))
        if image is None or image.size == 0:
            raise DetectorDatasetError("warehouse plate review queue image cannot be decoded")
        _verify_provenance(provenance[review_id], review_id, image_sha256)
        image_hashes.add(image_sha256)
    return len(queue)


def _verify_statistics(
    manifest: dict[str, Any],
    queue_count: int,
    duplicate_records: list[dict[str, Any]],
    reject_count: int,
) -> None:
    reasons = {
        reason: sum(record.get("reason") == reason for record in duplicate_records)
        for reason in (
            "EXACT_SHA256_DUPLICATE",
            "PERCEPTUAL_NEAR_DUPLICATE",
            "POST_CLEAN_EXACT_DUPLICATE",
        )
    }
    statistics = manifest.get("statistics")
    if (
        not isinstance(statistics, dict)
        or manifest.get("sampleCount") != 0
        or manifest.get("annotationCount") != 0
        or manifest.get("negativeSampleCount") != 0
        or manifest.get("suggestionCount") != 0
        or manifest.get("reviewQueueCount") != queue_count
        or statistics.get("uniqueReviewImages") != queue_count
        or statistics.get("exactDuplicateFilesExcluded") != reasons["EXACT_SHA256_DUPLICATE"]
        or statistics.get("perceptualNearDuplicateImagesExcluded")
        != reasons["PERCEPTUAL_NEAR_DUPLICATE"]
        or statistics.get("postCleanExactDuplicateImagesExcluded")
        != reasons["POST_CLEAN_EXACT_DUPLICATE"]
        or statistics.get("rejectedUniqueImages") != reject_count
    ):
        raise DetectorDatasetError("warehouse plate review source statistics do not match evidence")


def _provenance_record(
    *,
    review_id: str,
    image_sha256: str,
    archive_sha256: str,
    candidate: WarehouseImageCandidate,
    scan: WarehouseArchiveScan,
) -> dict[str, Any]:
    records = []
    for member in scan.names_by_digest[candidate.raw_sha256]:
        records.append(
            {
                "archiveMember": member.name,
                "sourceFilenameSha256": _sha256(member.name.encode("utf-8")),
                "sourceRawImageSha256": candidate.raw_sha256,
                "sourceArchiveSha256": archive_sha256,
                "cameraView": member.view,
                "cameraId": f"warehouse-{member.view}-camera",
                "groupId": f"phins-warehouse:{member.group_id}",
                "capturedAt": _timestamp(
                    datetime.fromtimestamp(member.timestamp_ms / 1000, tz=UTC)
                ),
                "recoveredVehicleBbox": list(candidate.bbox),
                "bboxRecoveryMethod": "BURNED_IN_BLUE_RECTANGLE",
                "bboxLineCoverage": round(candidate.bbox_coverage, 6),
                "imageBrightness": round(candidate.brightness, 4),
                "imageContrast": round(candidate.contrast, 4),
                "imageSharpness": round(candidate.sharpness, 4),
            }
        )
    return {
        "schemaVersion": 1,
        "reviewId": review_id,
        "sourceImageSha256": image_sha256,
        "records": records,
    }


def _verify_files(root: Path, entries: list[object]) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise DetectorDatasetError("warehouse plate review source file entry is invalid")
        relative = str(entry["path"])
        path = _safe_child(root, relative)
        if relative in files or not path.is_file() or path.is_symlink():
            raise DetectorDatasetError(
                "warehouse plate review source file is missing or duplicated"
            )
        if path.stat().st_size != int(entry.get("size", -1)):
            raise DetectorDatasetError(
                "warehouse plate review source file size verification failed"
            )
        if _sha256_file(path) != entry.get("sha256"):
            raise DetectorDatasetError("warehouse plate review source checksum verification failed")
        files[relative] = entry
    return files


def _verify_provenance(record: dict[str, Any], review_id: str, image_sha256: str) -> None:
    records = record.get("records") if isinstance(record, dict) else None
    if (
        record.get("schemaVersion") != 1
        or record.get("reviewId") != review_id
        or record.get("sourceImageSha256") != image_sha256
        or not isinstance(records, list)
        or not records
    ):
        raise DetectorDatasetError("warehouse plate review provenance is invalid")
    for item in records:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("archiveMember"), str)
            or not isinstance(item.get("sourceFilenameSha256"), str)
            or not _SHA256.fullmatch(str(item["sourceFilenameSha256"]))
            or not isinstance(item.get("sourceRawImageSha256"), str)
            or not _SHA256.fullmatch(str(item["sourceRawImageSha256"]))
            or not isinstance(item.get("sourceArchiveSha256"), str)
            or not _SHA256.fullmatch(str(item["sourceArchiveSha256"]))
        ):
            raise DetectorDatasetError("warehouse plate review provenance record is invalid")


def _read_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for record in _read_json_records(path):
        value = record.get(key)
        if not isinstance(value, str) or value in result:
            raise DetectorDatasetError(f"warehouse review evidence keys are invalid: {path.name}")
        result[value] = record
    return result


def _read_json_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_bytes().splitlines()
    except OSError as exc:
        raise DetectorDatasetError(f"cannot read warehouse review evidence: {path.name}") from exc
    for line in lines:
        if not line:
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DetectorDatasetError(
                f"warehouse review evidence is invalid: {path.name}"
            ) from exc
        if not isinstance(record, dict):
            raise DetectorDatasetError(f"warehouse review evidence record is invalid: {path.name}")
        records.append(record)
    return records


def _result(
    directory: Path,
    manifest: dict[str, Any],
    digest: str,
    *,
    reused: bool = False,
) -> WarehousePlateReviewSourceResult:
    source_archive = manifest["sourceArchive"]
    statistics = manifest["statistics"]
    return WarehousePlateReviewSourceResult(
        source_id=str(manifest["sourceId"]),
        directory=directory,
        manifest_sha256=digest,
        archive_sha256=str(source_archive["sha256"]),
        archive_image_count=int(source_archive["imageCount"]),
        unique_raw_image_count=int(statistics["uniqueRawImages"]),
        review_queue_count=int(manifest["reviewQueueCount"]),
        exact_duplicate_files_excluded=int(statistics["exactDuplicateFilesExcluded"]),
        near_duplicate_images_excluded=int(statistics["perceptualNearDuplicateImagesExcluded"]),
        post_clean_duplicate_images_excluded=int(
            statistics["postCleanExactDuplicateImagesExcluded"]
        ),
        rejected_unique_images=int(statistics["rejectedUniqueImages"]),
        reused=reused,
    )


def _source_card(
    *,
    source_id: str,
    review_count: int,
    exact_duplicates: int,
    near_duplicates: int,
    post_clean_duplicates: int,
    rejected: int,
) -> str:
    return f"""# {source_id}

Immutable review-only plate source staged from warehouse camera images.

- Cleaned images pending human plate review: {review_count}
- Exact duplicate files excluded: {exact_duplicates}
- Perceptual near-duplicate images excluded: {near_duplicates}
- Post-clean exact duplicates excluded: {post_clean_duplicates}
- Unrecoverable images rejected: {rejected}
- Rights review: required before production promotion
- Acceptance/release/distribution eligibility: disabled

The burned-in blue vehicle rectangle was recovered and inpainted. Perceptual
deduplication was applied to every recoverable frame before review. No vehicle
rectangle is treated as a plate label, and no image enters training until a
human approves/corrects a `license_plate` annotation or marks it negative.
"""


def _file_entry(path: Path, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": _sha256_file(path),
        "size": path.stat().st_size,
    }


def _safe_child(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or not posix.parts or ".." in posix.parts:
        raise DetectorDatasetError("warehouse plate review path is unsafe")
    path = root.joinpath(*posix.parts).resolve()
    if not path.is_relative_to(root):
        raise DetectorDatasetError("warehouse plate review path escapes its source")
    return path


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise DetectorDatasetError("warehouse plate review evidence already exists") from exc


def _remove_temporary(path: Path, expected_parent: Path) -> None:
    if path.exists() and path.parent == expected_parent and path.name.startswith("."):
        shutil.rmtree(path)


def _required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 256 or "\x00" in normalized:
        raise ValueError(f"{label} is invalid")
    return normalized


def _json_lines(records: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(record, pretty=False) for record in records)


def _json_bytes(value: object, *, pretty: bool) -> bytes:
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


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
