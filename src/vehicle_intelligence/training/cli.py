"""CLI composition root for offline detector model lifecycle operations."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vehicle_intelligence.application.dataset_review import DetectorReviewQuery
from vehicle_intelligence.config import DetectorConfig, VehicleDetectorConfig, load_settings
from vehicle_intelligence.domain.dataset_review import DetectorReviewDecision
from vehicle_intelligence.exceptions import (
    ModelEvaluationError,
    ModelRegistryError,
    VehicleIntelligenceError,
)
from vehicle_intelligence.infrastructure.training.dataset_review_files import (
    FileDetectorReviewRepository,
)
from vehicle_intelligence.infrastructure.vision.factory import (
    create_plate_detector,
    create_vehicle_detector,
)
from vehicle_intelligence.training.artifacts import (
    package_detector_candidate,
    verify_model_package,
)
from vehicle_intelligence.training.bootstrap import (
    BootstrapSourceWriter,
    acquire_bootstrap_samples,
    verify_bootstrap_source,
)
from vehicle_intelligence.training.bootstrap.http import BoundedHttpClient
from vehicle_intelligence.training.config import load_training_settings
from vehicle_intelligence.training.corpus import (
    VietnamPlateCorpusBuilder,
    verify_plate_corpus,
)
from vehicle_intelligence.training.dataset import (
    DetectorDatasetBuilder,
    verify_detector_dataset,
)
from vehicle_intelligence.training.domain import DatasetSplit, DetectorRole
from vehicle_intelligence.training.evaluation import (
    detector_release_gate_failures,
    evaluate_detector_files,
    evaluation_to_jsonable,
)
from vehicle_intelligence.training.first_party import (
    FirstPartyPlateSourceBuilder,
    verify_first_party_detector_source,
)
from vehicle_intelligence.training.huggingface import (
    HuggingFaceJobRunner,
    HuggingFacePrivateRegistry,
)
from vehicle_intelligence.training.inference import predict_dataset_split
from vehicle_intelligence.training.paddledetection import PaddleDetectionTrainer
from vehicle_intelligence.training.review_suggestions import (
    DetectorReviewSuggestionGenerator,
    ReviewSuggestionModel,
    ReviewSuggestionOptions,
)
from vehicle_intelligence.training.roboflow import (
    import_known_roboflow_archives,
    verify_roboflow_source,
)
from vehicle_intelligence.training.video_extraction import (
    VideoExtractionOptions,
    VideoTrainingImageExtractor,
)
from vehicle_intelligence.training.video_review_promotion import (
    AttestedVideoReviewPromotionBuilder,
)
from vehicle_intelligence.training.video_review_source import (
    VideoPlateReviewSourceBuilder,
    verify_video_plate_review_source,
)

_DEFAULT_CONFIG = Path("configs/model-training.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vehicle/plate detector model lifecycle")
    parser.add_argument("--config", type=Path, default=_DEFAULT_CONFIG)
    commands = parser.add_subparsers(dest="command_name", required=True)

    bootstrap = commands.add_parser(
        "bootstrap-samples",
        help="Acquire license-traced external samples for smoke/fine-tuning only",
    )
    _role_argument(bootstrap)
    bootstrap.add_argument("--samples-per-class", type=int, default=20)
    bootstrap.add_argument("--output", type=Path)

    ingest_corpus = commands.add_parser(
        "ingest-plate-corpus",
        help="Import the pinned VN polygon archive into the founder-namespaced corpus",
    )
    ingest_corpus.add_argument("archive", type=Path)

    ingest_roboflow = commands.add_parser(
        "ingest-roboflow-plate-archives",
        help="Import pinned Roboflow plate archives into canonical PHINS sources",
    )
    ingest_roboflow.add_argument("archives", type=Path, nargs="+")

    first_party = commands.add_parser(
        "ingest-first-party-plate-images",
        help="Build an immutable production source from user-collected plate images",
    )
    first_party.add_argument("input", type=Path)
    first_party.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/source/plate-first-party/phins-vn-plate-production-source-v1"
        ),
    )
    first_party.add_argument(
        "--source-id",
        default="phins-vn-plate-production-source-v1",
    )
    first_party.add_argument(
        "--label-reference",
        type=Path,
        default=Path("datasets/corpora/plate/phins-vn-plate-corpus-v2"),
    )
    first_party.add_argument(
        "--auto-reference",
        type=Path,
        default=Path("samples/extract/plate"),
    )

    verify_corpus = commands.add_parser("verify-corpus")
    verify_corpus.add_argument("corpus", type=Path)

    verify_roboflow = commands.add_parser("verify-roboflow-source")
    verify_roboflow.add_argument("source", type=Path)

    verify_source = commands.add_parser("verify-source")
    verify_source.add_argument("source", type=Path)

    verify_first_party = commands.add_parser("verify-first-party-source")
    verify_first_party.add_argument("source", type=Path)

    verify_video_review = commands.add_parser("verify-video-review-source")
    verify_video_review.add_argument("source", type=Path)

    extract_videos = commands.add_parser(
        "extract-video-samples",
        help="Extract reviewable vehicle and plate suggestions from local videos",
    )
    extract_videos.add_argument("input", type=Path)
    extract_videos.add_argument("--output", type=Path, default=Path("samples/extract"))
    extract_videos.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    extract_videos.add_argument(
        "--vehicle-model",
        type=Path,
        default=Path("models/yolo11n.pt"),
    )
    extract_videos.add_argument(
        "--plate-model",
        type=Path,
        default=Path("models/vietnam-plate.pt"),
    )
    extract_videos.add_argument("--device")
    extract_videos.add_argument("--sample-interval-seconds", type=float, default=1.0)
    extract_videos.add_argument("--vehicle-confidence", type=float, default=0.20)
    extract_videos.add_argument("--plate-confidence", type=float, default=0.15)
    extract_videos.add_argument("--detector-frame-max-edge", type=int, default=1920)
    extract_videos.add_argument("--plate-context-max-edge", type=int, default=1920)
    extract_videos.add_argument("--batch-size", type=int, default=8)
    extract_videos.add_argument("--maximum-vehicles-per-frame", type=int, default=24)
    extract_videos.add_argument("--maximum-plate-contexts-per-frame", type=int, default=12)

    stage_video_reviews = commands.add_parser(
        "stage-video-plate-review-source",
        help="Package extracted plate suggestions for the Dataset Review UI",
    )
    stage_video_reviews.add_argument("input", type=Path)
    stage_video_reviews.add_argument(
        "--source-id",
        default="phins-video-plate-review-v1",
    )
    stage_video_reviews.add_argument("--output", type=Path)

    promote_video_reviews = commands.add_parser(
        "promote-attested-video-review",
        help="Create a production source from a fully reviewed first-party video source",
    )
    promote_video_reviews.add_argument("source_id")
    promote_video_reviews.add_argument(
        "--base-source",
        type=Path,
        default=Path(
            "datasets/source/plate-first-party/phins-vn-plate-production-source-v2"
        ),
    )
    promote_video_reviews.add_argument(
        "--target-source-id",
        default="phins-vn-plate-production-source-v3",
    )
    promote_video_reviews.add_argument("--output", type=Path)
    promote_video_reviews.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    promote_video_reviews.add_argument("--rights-holder")
    promote_video_reviews.add_argument("--attested-by")
    promote_video_reviews.add_argument(
        "--confirm-first-party-rights",
        action="store_true",
        help="Explicitly attest that the video was first-party collected",
    )

    suggest_reviews = commands.add_parser(
        "suggest-review-labels",
        help="Add model suggestion overlays to unresolved detector-review images",
    )
    suggest_reviews.add_argument("source", type=Path)
    suggest_reviews.add_argument(
        "--runtime-config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    suggest_reviews.add_argument("--plate-model", type=Path, required=True)
    suggest_reviews.add_argument("--provider")
    suggest_reviews.add_argument("--model-name")
    suggest_reviews.add_argument("--model-version")
    suggest_reviews.add_argument("--device")
    suggest_reviews.add_argument("--image-size", type=int, default=1280)
    suggest_reviews.add_argument("--confidence", type=float, default=0.70)
    suggest_reviews.add_argument("--iou", type=float, default=0.60)
    suggest_reviews.add_argument("--batch-size", type=int, default=4)
    suggest_reviews.add_argument("--maximum-suggestions-per-image", type=int, default=4)
    suggest_reviews.add_argument("--limit", type=int)

    build = commands.add_parser("build-dataset", help="Build an immutable grouped COCO dataset")
    _role_argument(build)
    build.add_argument("--export-id", required=True)

    verify_dataset = commands.add_parser("verify-dataset")
    verify_dataset.add_argument("dataset", type=Path)

    train = commands.add_parser("train", help="Run configured PaddleDetection training")
    _role_argument(train)
    train.add_argument("dataset", type=Path)
    train.add_argument("--run-id", required=True)
    train.add_argument("--dry-run", action="store_true")

    export = commands.add_parser("export-onnx", help="Export trained Paddle weights to ONNX")
    _role_argument(export)
    export.add_argument("dataset", type=Path)
    export.add_argument("--weights", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)

    predict = commands.add_parser("predict", help="Create checksum-traced COCO predictions")
    _role_argument(predict)
    predict.add_argument("dataset", type=Path)
    predict.add_argument("--split", type=DatasetSplit, default=DatasetSplit.TEST)
    predict.add_argument("--runtime-config", type=Path, default=Path("configs/default.yaml"))
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--provider")
    predict.add_argument("--model", type=Path)
    predict.add_argument("--model-name")
    predict.add_argument("--model-version")
    predict.add_argument("--model-hash")
    predict.add_argument("--device")
    predict.add_argument("--image-size", type=int)
    predict.add_argument("--confidence", type=float)
    predict.add_argument("--iou", type=float)
    predict.add_argument("--model-classes", nargs="+")

    evaluate = commands.add_parser("evaluate", help="Evaluate and enforce detector gates")
    _role_argument(evaluate)
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument("--split", type=DatasetSplit, default=DatasetSplit.TEST)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--iou-threshold", type=float, default=0.5)
    evaluate.add_argument("--confidence-threshold", type=float, default=0.25)
    evaluate.add_argument("--full-bbox-coverage-threshold", type=float, default=0.95)

    package = commands.add_parser("package", help="Package a gate-passed ONNX candidate")
    _role_argument(package)
    package.add_argument("dataset", type=Path)
    package.add_argument("--onnx", type=Path, required=True)
    package.add_argument("--evaluation", type=Path, required=True)
    package.add_argument("--training-run", type=Path)
    package.add_argument("--model-name", required=True)
    package.add_argument("--model-version", required=True)
    package.add_argument("--output", type=Path, required=True)

    verify_package = commands.add_parser("verify-package")
    verify_package.add_argument("package", type=Path)

    upload_dataset = commands.add_parser("hf-upload-dataset")
    _role_argument(upload_dataset)
    upload_dataset.add_argument("dataset", type=Path)
    upload_dataset.add_argument("--revision", default="main")

    upload_model = commands.add_parser("hf-upload-model")
    _role_argument(upload_model)
    upload_model.add_argument("package", type=Path)
    upload_model.add_argument("--revision", default="main")

    job = commands.add_parser("hf-submit-job", help="Submit a custom private HF training job")
    _role_argument(job)
    job.add_argument("--image")
    job.add_argument("--flavor")
    job.add_argument("--timeout-seconds", type=int, default=86_400)
    job.add_argument("--name")
    job.add_argument("--env-from-local", action="append", default=[])
    job.add_argument("--secret-from-local", action="append", default=[])
    job.add_argument("job_command", nargs=argparse.REMAINDER)
    return parser


def run(args: argparse.Namespace) -> int:
    if args.command_name == "verify-source":
        manifest, digest = verify_bootstrap_source(args.source)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-first-party-source":
        manifest, digest = verify_first_party_detector_source(args.source)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-video-review-source":
        manifest, digest = verify_video_plate_review_source(args.source)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-corpus":
        manifest, digest = verify_plate_corpus(args.corpus)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-roboflow-source":
        manifest, digest = verify_roboflow_source(args.source)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-dataset":
        manifest, digest = verify_detector_dataset(args.dataset)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0
    if args.command_name == "verify-package":
        manifest, digest = verify_model_package(args.package)
        _print({"manifestSha256": digest, "manifest": manifest})
        return 0

    settings = load_training_settings(args.config)
    if args.command_name == "ingest-first-party-plate-images":
        result = FirstPartyPlateSourceBuilder(
            input_directory=args.input,
            output_directory=args.output,
            label_reference_directory=args.label_reference,
            auto_reference_directory=args.auto_reference,
            source_id=args.source_id,
            owner_namespace=settings.corpus.owner_namespace,
            founder_id=settings.corpus.founder_id,
        ).build()
        _print(
            {
                "sourceId": result.source_id,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sampleCount": result.sample_count,
                "annotationCount": result.annotation_count,
                "negativeSampleCount": result.negative_sample_count,
                "reviewQueueCount": result.review_queue_count,
                "exactDuplicateFilesExcluded": result.exact_duplicate_files_excluded,
                "unsupportedFileCount": result.unsupported_file_count,
                "reused": result.reused,
            }
        )
        return 0
    if args.command_name == "extract-video-samples":
        return _extract_video_samples(args, settings)
    if args.command_name == "stage-video-plate-review-source":
        output = args.output or (
            Path("datasets/source/plate-first-party") / args.source_id
        )
        result = VideoPlateReviewSourceBuilder(
            extraction_directory=args.input,
            output_directory=output,
            source_id=args.source_id,
            owner_namespace=settings.corpus.owner_namespace,
            founder_id=settings.corpus.founder_id,
        ).build()
        _print(
            {
                "sourceId": result.source_id,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sourceRecordCount": result.source_record_count,
                "reviewQueueCount": result.review_queue_count,
                "suggestionCount": result.suggestion_count,
                "exactDuplicateImagesMerged": result.exact_duplicate_images_merged,
                "promotionEligible": False,
                "releaseEligible": False,
                "reused": result.reused,
            }
        )
        return 0
    if args.command_name == "promote-attested-video-review":
        if not args.confirm_first_party_rights:
            raise ModelRegistryError(
                "attested video promotion requires --confirm-first-party-rights"
            )
        runtime_settings = load_settings(args.runtime_config)
        review_source, decisions = asyncio.run(
            _completed_review_decisions(runtime_settings, args.source_id)
        )
        output = args.output or (
            runtime_settings.dataset_review.promoted_sources_directory
            / args.target_source_id
        )
        rights_holder = args.rights_holder or settings.corpus.founder_id
        attested_by = args.attested_by or settings.corpus.founder_id
        result = AttestedVideoReviewPromotionBuilder(
            base_source_directory=args.base_source,
            review_source_directory=review_source,
            output_directory=output,
            target_source_id=args.target_source_id,
            decisions=decisions,
            rights_holder=rights_holder,
            attested_by=attested_by,
        ).build()
        _print(
            {
                "sourceId": result.source_id,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sampleCount": result.sample_count,
                "annotationCount": result.annotation_count,
                "negativeSampleCount": result.negative_sample_count,
                "promotedReviewCount": result.promoted_review_count,
                "promotedPositiveCount": result.promoted_positive_count,
                "promotedNegativeCount": result.promoted_negative_count,
                "rejectedCount": result.rejected_count,
                "releaseEligible": True,
                "distributionEligible": False,
                "reused": result.reused,
            }
        )
        return 0
    if args.command_name == "suggest-review-labels":
        return _suggest_review_labels(args)
    if args.command_name == "ingest-roboflow-plate-archives":
        results = import_known_roboflow_archives(
            args.archives,
            owner_namespace=settings.corpus.owner_namespace,
            founder_id=settings.corpus.founder_id,
            detection_output_root=settings.corpus.plate_external_sources_directory,
            auxiliary_output_root=settings.corpus.plate_auxiliary_output_directory,
        )
        _print(
            {
                "sources": [
                    {
                        "sourceId": result.source_id,
                        "task": result.task,
                        "directory": str(result.directory),
                        "manifestSha256": result.manifest_sha256,
                        "sourceImageCount": result.source_image_count,
                        "canonicalImageCount": result.canonical_image_count,
                        "annotationCount": result.annotation_count,
                        "negativeSampleCount": result.negative_sample_count,
                        "duplicateImagesMerged": result.duplicate_images_merged,
                        "reused": result.reused,
                    }
                    for result in results
                ]
            }
        )
        return 0
    if args.command_name == "ingest-plate-corpus":
        result = VietnamPlateCorpusBuilder(settings.corpus).build(args.archive)
        _print(
            {
                "corpusId": result.corpus_id,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sampleCount": result.sample_count,
                "annotationCount": result.annotation_count,
                "duplicateImagesMerged": result.duplicate_images_merged,
                "acceptanceEligible": False,
                "releaseEligible": False,
                "distributionEligible": False,
                "reused": result.reused,
            }
        )
        return 0
    role = DetectorRole(args.role)
    target = settings.target(role)

    if args.command_name == "build-dataset":
        result = DetectorDatasetBuilder(target.dataset).build(args.export_id)
        _print(
            {
                "exportId": result.export_id,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sampleCount": result.sample_count,
                "annotationCount": result.annotation_count,
                "splitCounts": result.split_counts,
                "reused": result.reused,
            }
        )
        return 0
    if args.command_name == "bootstrap-samples":
        with BoundedHttpClient() as http:
            source, samples = acquire_bootstrap_samples(
                role,
                samples_per_class=args.samples_per_class,
                http=http,
            )
        bootstrap_output = args.output or Path("datasets/source") / role.value
        result = BootstrapSourceWriter(role, bootstrap_output).write(
            source,
            samples,
        )
        _print(
            {
                "role": result.role.value,
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "sampleCount": result.sample_count,
                "annotationCount": result.annotation_count,
                "acceptanceEligible": False,
                "reused": result.reused,
            }
        )
        return 0

    trainer = PaddleDetectionTrainer(
        target.paddledetection,
        role,
        target.dataset.classes,
    )
    if args.command_name == "train":
        if args.dry_run:
            run_directory = target.paddledetection.output_directory / args.run_id
            _print({"command": list(trainer.build_train_command(args.dataset, run_directory))})
            return 0
        result = trainer.train(args.dataset, args.run_id)
        _print(result.model_dump(mode="json"))
        return 0
    if args.command_name == "export-onnx":
        output = trainer.export_onnx(args.dataset, args.weights, args.output)
        _print({"onnx": str(output), "sha256": _sha256_file(output)})
        return 0
    if args.command_name == "predict":
        return _predict(args, role, target.dataset.classes)
    if args.command_name == "evaluate":
        return _evaluate(args, role, target.gates)
    if args.command_name == "package":
        result = package_detector_candidate(
            role=role,
            model_name=args.model_name,
            model_version=args.model_version,
            classes=target.dataset.classes,
            onnx_path=args.onnx,
            dataset_directory=args.dataset,
            evaluation_path=args.evaluation,
            output_directory=args.output,
            training_run_path=args.training_run,
        )
        _print(
            {
                "directory": str(result.directory),
                "manifestSha256": result.manifest_sha256,
                "modelSha256": result.model_sha256,
                "reused": result.reused,
            }
        )
        return 0
    if (
        args.command_name in {"hf-upload-dataset", "hf-upload-model", "hf-submit-job"}
        and not settings.huggingface.enabled
    ):
        raise ModelRegistryError("Hugging Face integration is disabled in training config")
    if args.command_name == "hf-upload-dataset":
        result = HuggingFacePrivateRegistry().upload_dataset(
            args.dataset,
            target.hub.dataset_repo,
            revision=args.revision,
        )
        _print(result.model_dump(mode="json"))
        return 0
    if args.command_name == "hf-upload-model":
        result = HuggingFacePrivateRegistry().upload_model(
            args.package,
            target.hub.model_repo,
            revision=args.revision,
        )
        _print(result.model_dump(mode="json"))
        return 0
    if args.command_name == "hf-submit-job":
        if not settings.huggingface.jobs_enabled:
            raise ModelRegistryError("Hugging Face Jobs are disabled in training config")
        command = list(args.job_command)
        if command and command[0] == "--":
            command = command[1:]
        result = HuggingFaceJobRunner().submit(
            image=args.image or settings.huggingface.job_image,
            command=command,
            flavor=args.flavor or settings.huggingface.job_flavor,
            dataset_repo=target.hub.dataset_repo,
            output_bucket=settings.huggingface.job_output_bucket or "",
            namespace=settings.huggingface.job_namespace,
            timeout_seconds=args.timeout_seconds,
            environment=_local_environment(args.env_from_local, secret=False),
            secrets=_local_environment(args.secret_from_local, secret=True),
            name=args.name,
        )
        _print(result.model_dump(mode="json"))
        return 0
    raise AssertionError(f"unhandled model training command: {args.command_name}")


def _extract_video_samples(args: argparse.Namespace, training_settings: Any) -> int:
    runtime_settings = load_settings(args.runtime_config)
    vehicle_model = args.vehicle_model.expanduser().resolve()
    plate_model = args.plate_model.expanduser().resolve()
    vehicle_config = VehicleDetectorConfig.model_validate(
        {
            **runtime_settings.vision.vehicle_detection.model_dump(),
            "model_path": str(args.vehicle_model.expanduser().resolve()),
            "model_hash": _sha256_file(vehicle_model),
            "confidence": args.vehicle_confidence,
            "device": args.device,
        }
    )
    plate_config = DetectorConfig.model_validate(
        {
            **runtime_settings.vision.plate_detection.model_dump(),
            "model_path": str(plate_model),
            "model_hash": _sha256_file(plate_model),
            "confidence": args.plate_confidence,
            "device": args.device,
        }
    )
    options = VideoExtractionOptions(
        input_directory=args.input,
        output_directory=args.output,
        sample_interval_seconds=args.sample_interval_seconds,
        detector_frame_max_edge=args.detector_frame_max_edge,
        plate_context_max_edge=args.plate_context_max_edge,
        batch_size=args.batch_size,
        maximum_vehicles_per_frame=args.maximum_vehicles_per_frame,
        maximum_plate_contexts_per_frame=args.maximum_plate_contexts_per_frame,
    )

    def progress(event: str, payload: dict[str, Any]) -> None:
        print(
            json.dumps({"event": event, **payload}, sort_keys=True, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    result = VideoTrainingImageExtractor(
        create_vehicle_detector(vehicle_config),
        create_plate_detector(plate_config),
        options,
        owner_namespace=training_settings.corpus.owner_namespace,
        founder_id=training_settings.corpus.founder_id,
        model_evidence={
            "vehicle": {
                "provider": vehicle_config.provider,
                "name": vehicle_config.model_name,
                "version": vehicle_config.model_version,
                "sha256": vehicle_config.model_hash,
                "artifact": vehicle_model.name,
                "confidence": vehicle_config.confidence,
                "iou": vehicle_config.iou,
            },
            "plate": {
                "provider": plate_config.provider,
                "name": plate_config.model_name,
                "version": plate_config.model_version,
                "sha256": plate_config.model_hash,
                "artifact": plate_model.name,
                "confidence": plate_config.confidence,
                "iou": plate_config.iou,
            },
        },
        progress=progress,
    ).extract()
    _print(
        {
            "outputDirectory": str(result.output_directory),
            "manifest": str(result.manifest_path),
            "videosDiscovered": result.videos_discovered,
            "videosProcessed": result.videos_processed,
            "sampledFrames": result.sampled_frames,
            "vehicleTrainingImages": result.vehicle_training_images,
            "vehicleCropImages": result.vehicle_crop_images,
            "plateTrainingImages": result.plate_training_images,
            "plateCropImages": result.plate_crop_images,
            "vehicleClassCounts": result.vehicle_class_counts,
            "failedVideos": list(result.failed_videos),
        }
    )
    return 0 if not result.failed_videos else 3


async def _completed_review_decisions(
    runtime_settings: Any,
    source_id: str,
) -> tuple[Path, dict[str, DetectorReviewDecision]]:
    config = runtime_settings.dataset_review
    repository = FileDetectorReviewRepository(config)
    await repository.initialize()
    try:
        summaries = await repository.list_sources()
        summary = next((item for item in summaries if item.source_id == source_id), None)
        if summary is None:
            raise ModelRegistryError(f"video review source not found: {source_id}")
        if summary.pending_count or summary.reviewed_count != summary.queue_count:
            raise ModelRegistryError(
                "video review source must have a terminal decision for every queue item"
            )
        decisions: dict[str, DetectorReviewDecision] = {}
        cursor: str | None = None
        while True:
            page = await repository.list_items(
                DetectorReviewQuery(
                    source_id=source_id,
                    limit=200,
                    cursor=cursor,
                )
            )
            for item in page.items:
                history = await repository.decision_history(source_id, item.review_id)
                if not history:
                    raise ModelRegistryError(
                        f"review decision history is missing: {item.review_id}"
                    )
                decisions[item.review_id] = history[-1]
            cursor = page.next_cursor
            if cursor is None:
                break
        if len(decisions) != summary.queue_count:
            raise ModelRegistryError("video review decision snapshot is incomplete")
        source = (config.sources_directory / source_id).expanduser().resolve()
        return source, decisions
    finally:
        await repository.close()


def _suggest_review_labels(args: argparse.Namespace) -> int:
    runtime_settings = load_settings(args.runtime_config)
    model_path = args.plate_model.expanduser().resolve()
    model_hash = _sha256_file(model_path)
    base = runtime_settings.vision.plate_detection.model_dump()
    updates: dict[str, object] = {
        "model_path": str(model_path),
        "model_hash": model_hash,
        "confidence": args.confidence,
        "iou": args.iou,
        "image_size": args.image_size,
    }
    for key, value in {
        "provider": args.provider,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "device": args.device,
    }.items():
        if value is not None:
            updates[key] = value
    detector_config = DetectorConfig.model_validate({**base, **updates})
    review_config = runtime_settings.dataset_review

    def progress(event: str, payload: dict[str, object]) -> None:
        print(
            json.dumps({"event": event, **payload}, sort_keys=True, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )

    result = DetectorReviewSuggestionGenerator(
        create_plate_detector(detector_config),
        ReviewSuggestionModel(
            provider=detector_config.provider,
            name=detector_config.model_name,
            version=detector_config.model_version,
            sha256=model_hash,
            confidence=detector_config.confidence,
            iou=detector_config.iou,
            image_size=detector_config.image_size,
        ),
        ReviewSuggestionOptions(
            source_directory=args.source,
            workspace_directory=review_config.workspace_directory,
            batch_size=args.batch_size,
            maximum_suggestions_per_image=args.maximum_suggestions_per_image,
            maximum_image_bytes=review_config.maximum_image_bytes,
            maximum_image_pixels=review_config.maximum_image_pixels,
            limit=args.limit,
        ),
        progress=progress,
    ).generate()
    _print(
        {
            "sourceId": result.source_id,
            "sourceManifestSha256": result.source_manifest_sha256,
            "suggestionRunId": result.suggestion_run_id,
            "candidates": result.candidates,
            "scanned": result.scanned,
            "suggestedItems": result.suggested_items,
            "suggestionBoxes": result.suggestion_boxes,
            "noDetection": result.no_detection,
            "skippedHumanReviewed": result.skipped_human_reviewed,
            "skippedSourceSuggestions": result.skipped_source_suggestions,
            "reusedEvidence": result.reused_evidence,
            "failureCount": len(result.failures),
            "failures": list(result.failures),
            "workspaceDirectory": str(result.workspace_directory),
        }
    )
    return 0 if not result.failures else 3


def _predict(
    args: argparse.Namespace,
    role: DetectorRole,
    target_classes: tuple[str, ...],
) -> int:
    settings = load_settings(args.runtime_config)
    if role is DetectorRole.VEHICLE:
        original_config: DetectorConfig = settings.vision.vehicle_detection
    else:
        original_config = settings.vision.plate_detection
    candidate_hash = (
        args.model_hash or _sha256_file(args.model.expanduser().resolve())
        if args.model is not None
        else args.model_hash
    )
    updates = {
        key: value
        for key, value in {
            "provider": args.provider,
            "model_path": str(args.model) if args.model is not None else None,
            "model_name": args.model_name,
            "model_version": args.model_version,
            "model_hash": candidate_hash,
            "device": args.device,
            "image_size": args.image_size,
            "confidence": args.confidence,
            "iou": args.iou,
            "model_classes": (
                args.model_classes
                if args.model_classes is not None
                else list(target_classes) if args.model is not None else None
            ),
        }.items()
        if value is not None
    }
    merged = original_config.model_copy(update=updates).model_dump()
    if not merged.get("model_path"):
        raise ModelEvaluationError("detector candidate model path is required")
    if role is DetectorRole.VEHICLE:
        vehicle_config = VehicleDetectorConfig.model_validate(merged)
        config: DetectorConfig = vehicle_config
        detector = create_vehicle_detector(vehicle_config)
    else:
        config = DetectorConfig.model_validate(merged)
        detector = create_plate_detector(config)
    model_info = {
        "provider": config.provider,
        "name": config.model_name,
        "version": config.model_version,
        "configuredHash": config.model_hash,
        "path": config.model_path,
    }
    result = predict_dataset_split(
        args.dataset,
        args.split,
        role,
        detector,
        args.output,
        model_info=model_info,
    )
    _print(
        {
            "predictions": str(result.predictions_path),
            "manifest": str(result.manifest_path),
            "imageCount": result.image_count,
            "predictionCount": result.prediction_count,
            "predictionsSha256": result.predictions_sha256,
        }
    )
    return 0


def _evaluate(args: argparse.Namespace, role: DetectorRole, gates: Any) -> int:
    dataset_manifest, dataset_digest = verify_detector_dataset(args.dataset)
    if dataset_manifest["role"] != role.value:
        raise ModelEvaluationError("evaluation role does not match detector dataset")
    predictions = args.predictions.expanduser().resolve()
    evidence_verified = _verify_prediction_manifest(
        predictions,
        dataset_digest,
        role,
        args.split,
    )
    annotations = args.dataset.expanduser().resolve() / "annotations" / f"{args.split.value}.json"
    evaluation = evaluate_detector_files(
        annotations,
        predictions,
        iou_threshold=args.iou_threshold,
        confidence_threshold=args.confidence_threshold,
        full_bbox_coverage_threshold=args.full_bbox_coverage_threshold,
    )
    failures = detector_release_gate_failures(evaluation, gates)
    payload = {
        "schemaVersion": 1,
        "type": "DETECTOR_EVALUATION",
        "createdAt": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "role": role.value,
        "split": args.split.value,
        "datasetExportId": dataset_manifest["exportId"],
        "datasetManifestSha256": dataset_digest,
        "predictionsSha256": _sha256_file(predictions),
        "evidenceVerified": evidence_verified,
        "metrics": evaluation_to_jsonable(evaluation),
        "releaseGate": {"passed": not failures and evidence_verified, "failures": failures},
    }
    if not evidence_verified:
        payload["releaseGate"]["failures"] = [*failures, "prediction_evidence"]
    output = args.output.expanduser().resolve()
    if output.exists():
        raise ModelEvaluationError("detector evaluation report already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_new(output, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())
    _print(payload)
    return 0 if payload["releaseGate"]["passed"] else 4


def _verify_prediction_manifest(
    predictions: Path,
    dataset_digest: str,
    role: DetectorRole,
    split: DatasetSplit,
) -> bool:
    manifest_path = predictions.with_suffix(predictions.suffix + ".manifest.json")
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(manifest, dict)
        and manifest.get("schemaVersion") == 1
        and manifest.get("type") == "DETECTOR_PREDICTIONS"
        and manifest.get("role") == role.value
        and manifest.get("split") == split.value
        and manifest.get("datasetManifestSha256") == dataset_digest
        and manifest.get("predictionsSha256") == _sha256_file(predictions)
    )


def _local_environment(names: list[str], *, secret: bool) -> dict[str, str]:
    values: dict[str, str] = {}
    for name in names:
        if not name or not name.replace("_", "A").isalnum() or name[0].isdigit():
            raise ModelRegistryError("Job environment variable name is invalid")
        value = os.environ.get(name)
        if value is None:
            kind = "secret" if secret else "environment variable"
            raise ModelRegistryError(f"local {kind} is missing: {name}")
        values[name] = value
    return values


def _role_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--role", type=DetectorRole, required=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _print(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except (VehicleIntelligenceError, OSError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
