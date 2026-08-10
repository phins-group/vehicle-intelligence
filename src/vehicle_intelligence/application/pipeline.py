"""Video-file application pipeline; no database or model SDK imports."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.direction import DirectionEstimator, point_in_polygon
from vehicle_intelligence.application.finalization import VehicleEventFinalizer
from vehicle_intelligence.application.health import CameraHealthReporter
from vehicle_intelligence.application.normalization import VietnamPlateNormalizer
from vehicle_intelligence.application.plate_crop import expanded_plate_detection
from vehicle_intelligence.application.ports import (
    LivePreviewSink,
    OCRProvider,
    PlateDetector,
    PlatePreprocessor,
    StreamHeartbeat,
    VehicleDetector,
    VehicleTracker,
    VideoSource,
)
from vehicle_intelligence.application.quality import PlateQualityEvaluator
from vehicle_intelligence.application.selection import BestFrameSelector
from vehicle_intelligence.config import Settings
from vehicle_intelligence.domain import (
    BoundingBox,
    Detection,
    ImageCandidate,
    LiveFrameMetadata,
    LivePlateOverlay,
    LiveVehicleOverlay,
    OCRResult,
    PlateDetection,
    PlateNormalization,
    PlateObservation,
    PlateQuality,
    Point,
    TrackedDetection,
    VehicleEvent,
    VehicleTrack,
    VideoFrame,
)
from vehicle_intelligence.exceptions import InferenceError

logger = logging.getLogger(__name__)
TrackKey = tuple[int, int]


@dataclass(slots=True)
class PipelineStats:
    decoded_frames: int = 0
    sampled_frames: int = 0
    vehicle_detections: int = 0
    plate_detections: int = 0
    quality_rejections: int = 0
    ocr_requests: int = 0
    ocr_observations: int = 0
    ocr_failures: int = 0
    finalized_tracks: int = 0
    vehicle_inference_calls: int = 0
    vehicle_inference_seconds: float = 0.0
    plate_inference_calls: int = 0
    plate_inference_seconds: float = 0.0
    ocr_inference_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class PipelineResult:
    events: tuple[VehicleEvent, ...]
    stats: PipelineStats


class VideoVehiclePipeline:
    def __init__(
        self,
        settings: Settings,
        source: VideoSource,
        vehicle_detector: VehicleDetector | None,
        tracker: VehicleTracker,
        plate_detector: PlateDetector,
        quality_evaluator: PlateQualityEvaluator,
        preprocessor: PlatePreprocessor,
        ocr: OCRProvider,
        normalizer: VietnamPlateNormalizer,
        selector: BestFrameSelector,
        finalizer: VehicleEventFinalizer,
        direction_estimator: DirectionEstimator,
        retain_events: bool = True,
        event_observer: Callable[[VehicleEvent], None] | None = None,
        health_reporter: CameraHealthReporter | None = None,
        live_preview: LivePreviewSink | None = None,
    ) -> None:
        if settings.vision.plate_only != (vehicle_detector is None):
            raise ValueError(
                "plate-only mode requires no vehicle detector; vehicle mode requires one"
            )
        self._settings = settings
        self._source = source
        self._vehicle_detector = vehicle_detector
        self._tracker = tracker
        self._plate_detector = plate_detector
        self._quality_evaluator = quality_evaluator
        self._preprocessor = preprocessor
        self._ocr = ocr
        self._normalizer = normalizer
        self._selector = selector
        self._finalizer = finalizer
        self._direction_estimator = direction_estimator
        self._retain_events = retain_events
        self._event_observer = event_observer
        self._health_reporter = health_reporter
        self._live_preview = live_preview
        self._active: dict[TrackKey, VehicleTrack] = {}
        self._completed_track_ids: dict[TrackKey, datetime] = {}
        self._stream_epoch: int | None = None
        self._stats = PipelineStats()
        self._started_at = time.monotonic()
        self._roi = (
            tuple(Point(x, y) for x, y in settings.camera.roi)
            if settings.camera.roi is not None
            else None
        )
        self._crossing_line = (
            tuple(Point(x, y) for x, y in settings.camera.crossing_line)
            if settings.camera.crossing_line is not None
            else None
        )

    async def run(self) -> PipelineResult:
        events: list[VehicleEvent] = []
        last_timestamp: datetime | None = None
        should_finalize = False
        try:
            for source_item in self._source.frames():
                await self._report_health()
                self._expire_completed_tracks(source_item.timestamp)
                self._record_events(
                    events,
                    await self._change_stream_epoch(source_item.stream_epoch),
                )
                if isinstance(source_item, StreamHeartbeat):
                    self._record_events(
                        events,
                        await self._finalize_timed_out(source_item.timestamp),
                    )
                    continue
                frame = source_item
                self._stats.sampled_frames += 1
                last_timestamp = frame.timestamp
                if self._settings.vision.plate_only:
                    frame_events, live_vehicles = await self._process_plate_only_frame(frame)
                else:
                    frame_events, live_vehicles = await self._process_vehicle_frame(frame)
                self._record_events(events, frame_events)
                await self._report_live_preview(frame, live_vehicles)
                self._record_events(events, await self._finalize_timed_out(frame.timestamp))
            should_finalize = True
        except KeyboardInterrupt:
            should_finalize = True
            logger.info(
                "pipeline_interrupted",
                extra={"camera_id": self._settings.camera.id},
            )
        except asyncio.CancelledError:
            should_finalize = True
            logger.info(
                "pipeline_cancelled",
                extra={"camera_id": self._settings.camera.id},
            )
            raise
        finally:
            try:
                if should_finalize:
                    self._record_events(events, await self._finalize_all())
            finally:
                self._source.close()
                await self._report_health(force=True)
                self._tracker.reset()
        if last_timestamp is None:
            logger.warning(
                "video_source_produced_no_frames",
                extra={"camera_id": self._settings.camera.id},
            )
        self._stats.decoded_frames = int(
            getattr(self._source, "decoded_frames", self._stats.sampled_frames)
        )
        return PipelineResult(tuple(events), self._stats)

    def _record_events(
        self,
        retained: list[VehicleEvent],
        emitted: tuple[VehicleEvent, ...] | list[VehicleEvent],
    ) -> None:
        for event in emitted:
            if self._event_observer is not None:
                try:
                    self._event_observer(event)
                except Exception:
                    logger.exception(
                        "event_observer_failed",
                        extra={"camera_id": event.camera.id, "event_id": event.id},
                    )
            if self._retain_events:
                retained.append(event)

    async def _process_vehicle_frame(
        self,
        frame: VideoFrame,
    ) -> tuple[list[VehicleEvent], list[LiveVehicleOverlay]]:
        if self._vehicle_detector is None:
            raise AssertionError("vehicle detector is missing outside plate-only mode")
        inference_started = time.perf_counter()
        detections = self._vehicle_detector.detect(frame.image)
        self._stats.vehicle_inference_seconds += time.perf_counter() - inference_started
        self._stats.vehicle_inference_calls += 1
        detections = [item for item in detections if self._inside_roi(item.bbox)]
        self._stats.vehicle_detections += len(detections)
        tracked = self._tracker.update(detections, frame.image)
        events: list[VehicleEvent] = []
        live_vehicles: list[LiveVehicleOverlay] = []
        for item in tracked:
            key = (frame.stream_epoch, item.track_id)
            if key in self._completed_track_ids:
                self._completed_track_ids[key] = frame.timestamp
                continue
            event, live_vehicle = await self._process_tracked(
                frame.image,
                frame.frame_id,
                frame.timestamp,
                frame.stream_epoch,
                item,
            )
            if live_vehicle is not None:
                live_vehicles.append(live_vehicle)
            if event is not None:
                events.append(event)
        return events, live_vehicles

    async def _process_plate_only_frame(
        self,
        frame: VideoFrame,
    ) -> tuple[list[VehicleEvent], list[LiveVehicleOverlay]]:
        plate_detections = [
            item
            for item in self._detect_plates(frame.image)
            if self._inside_roi(item.bbox)
        ]
        tracker_input = [
            Detection(
                bbox=item.bbox,
                confidence=item.confidence,
                class_id=0,
                class_name="plate",
                model=item.model,
            )
            for item in plate_detections
        ]
        tracked = self._tracker.update(tracker_input, frame.image)
        unmatched = list(plate_detections)
        events: list[VehicleEvent] = []
        live_plates: list[LiveVehicleOverlay] = []
        for item in tracked:
            detection = self._pop_matching_plate(unmatched, item.detection.bbox)
            if detection is None:
                continue
            key = (frame.stream_epoch, item.track_id)
            if key in self._completed_track_ids:
                self._completed_track_ids[key] = frame.timestamp
                continue
            event, overlay = await self._process_plate_only_tracked(
                frame,
                item,
                detection,
            )
            if overlay is not None:
                live_plates.append(overlay)
            if event is not None:
                events.append(event)
        return events, live_plates

    async def _process_plate_only_tracked(
        self,
        frame: VideoFrame,
        tracked: TrackedDetection,
        detection: PlateDetection,
    ) -> tuple[VehicleEvent | None, LiveVehicleOverlay | None]:
        height, width = frame.image.shape[:2]
        tracked_bbox = tracked.detection.bbox.clip(width, height)
        detection_bbox = detection.bbox.clip(width, height)
        if tracked_bbox is None or detection_bbox is None:
            return None, None
        if detection_bbox is not detection.bbox:
            detection = replace(detection, bbox=detection_bbox)
        key = (frame.stream_epoch, tracked.track_id)
        track = self._track_for(key, frame.timestamp, frame.stream_epoch, tracked.track_id)
        track.update_plate_track(
            detection,
            tracked_bbox,
            frame.frame_id,
            frame.timestamp,
        )
        plate_crop = self._crop(frame.image, detection_bbox)
        self._consider_plate_only_snapshot(
            track,
            frame.image,
            plate_crop,
            detection_bbox,
            detection.confidence,
        )
        plate = self._evaluate_plate_detections(
            track,
            frame.image,
            BoundingBox(0, 0, width, height),
            [detection],
            frame.frame_id,
            frame.timestamp,
        )
        direction = self._direction_estimator.estimate(tuple(track.trajectory))
        track.direction = direction
        overlay = LiveVehicleOverlay(
            track_id=track.logical_id,
            bbox=tracked_bbox,
            confidence=detection.confidence,
            vehicle_type="unknown",
            direction=direction,
            plate=plate,
        )
        if self._settings.camera.finalize_on_crossing and direction.value != "UNKNOWN":
            return await self._finalize_track(key), overlay
        return None, overlay

    async def _process_tracked(
        self,
        frame_image: NDArray[np.uint8],
        frame_id: int,
        timestamp: datetime,
        stream_epoch: int,
        tracked: TrackedDetection,
    ) -> tuple[VehicleEvent | None, LiveVehicleOverlay | None]:
        detection = tracked.detection
        clipped = detection.bbox.clip(frame_image.shape[1], frame_image.shape[0])
        if clipped is None:
            return None, None
        key = (stream_epoch, tracked.track_id)
        track = self._track_for(key, timestamp, stream_epoch, tracked.track_id)
        if clipped is not detection.bbox:
            tracked = TrackedDetection(
                tracked.track_id,
                type(detection)(
                    bbox=clipped,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    class_name=detection.class_name,
                    model=detection.model,
                ),
            )
        track.update(tracked, frame_id, timestamp)
        vehicle_crop = self._crop(frame_image, clipped)
        self._consider_vehicle_images(track, frame_image, vehicle_crop, clipped, tracked)
        plate = self._observe_plate(track, vehicle_crop, clipped, frame_id, timestamp)

        direction = self._direction_estimator.estimate(tuple(track.trajectory))
        track.direction = direction
        live_vehicle = LiveVehicleOverlay(
            track_id=track.logical_id,
            bbox=clipped,
            confidence=tracked.detection.confidence,
            vehicle_type=tracked.detection.class_name,
            direction=direction,
            plate=plate,
        )
        if self._settings.camera.finalize_on_crossing and direction.value != "UNKNOWN":
            return await self._finalize_track(key), live_vehicle
        return None, live_vehicle

    def _track_for(
        self,
        key: TrackKey,
        timestamp: datetime,
        stream_epoch: int,
        track_id: int,
    ) -> VehicleTrack:
        track = self._active.get(key)
        if track is not None:
            return track
        session_id = self._source.source_id
        if stream_epoch:
            session_id = f"{session_id}-e{stream_epoch}"
        track = VehicleTrack(
            camera_id=self._settings.camera.id,
            session_id=session_id,
            local_track_id=track_id,
            first_seen=timestamp,
            last_seen=timestamp,
            max_trajectory_points=self._settings.tracking.max_trajectory_points,
            max_plate_observations=self._settings.tracking.max_plate_observations,
        )
        self._active[key] = track
        return track

    async def _change_stream_epoch(self, stream_epoch: int) -> list[VehicleEvent]:
        if self._stream_epoch is None:
            self._stream_epoch = stream_epoch
            return []
        if stream_epoch == self._stream_epoch:
            return []
        previous = self._stream_epoch
        events = await self._finalize_all()
        self._tracker.reset()
        self._completed_track_ids.clear()
        self._stream_epoch = stream_epoch
        logger.info(
            "stream_epoch_changed",
            extra={
                "camera_id": self._settings.camera.id,
                "previous_epoch": previous,
                "stream_epoch": stream_epoch,
            },
        )
        return events

    def _consider_vehicle_images(
        self,
        track: VehicleTrack,
        frame: NDArray[np.uint8],
        vehicle_crop: NDArray[np.uint8],
        bbox: BoundingBox,
        tracked: TrackedDetection,
    ) -> None:
        score = self._selector.vehicle_score(
            vehicle_crop,
            bbox,
            frame.shape[1],
            frame.shape[0],
            tracked.detection.confidence,
        )
        if track.best_snapshot is None or score > track.best_snapshot.score:
            track.best_snapshot = ImageCandidate(
                frame_id=track.trajectory[-1].frame_id,
                timestamp=track.last_seen,
                score=score,
                image=frame.copy(),
            )
            track.best_vehicle_crop = ImageCandidate(
                frame_id=track.trajectory[-1].frame_id,
                timestamp=track.last_seen,
                score=score,
                image=vehicle_crop.copy(),
            )

    def _consider_plate_only_snapshot(
        self,
        track: VehicleTrack,
        frame: NDArray[np.uint8],
        plate_crop: NDArray[np.uint8],
        bbox: BoundingBox,
        detector_confidence: float,
    ) -> None:
        score = self._selector.vehicle_score(
            plate_crop,
            bbox,
            frame.shape[1],
            frame.shape[0],
            detector_confidence,
        )
        if track.best_snapshot is None or score > track.best_snapshot.score:
            track.best_snapshot = ImageCandidate(
                frame_id=track.trajectory[-1].frame_id,
                timestamp=track.last_seen,
                score=score,
                image=frame.copy(),
            )

    def _detect_plates(
        self,
        image: NDArray[np.uint8],
        track: VehicleTrack | None = None,
    ) -> list[PlateDetection]:
        inference_started = time.perf_counter()
        try:
            detections = self._plate_detector.detect(image)
        except InferenceError:
            context: dict[str, object] = {"camera_id": self._settings.camera.id}
            if track is not None:
                context["track_id"] = track.logical_id
            logger.exception("plate_detection_failed", extra=context)
            return []
        finally:
            self._stats.plate_inference_seconds += time.perf_counter() - inference_started
            self._stats.plate_inference_calls += 1
        self._stats.plate_detections += len(detections)
        return detections

    def _observe_plate(
        self,
        track: VehicleTrack,
        vehicle_crop: NDArray[np.uint8],
        vehicle_bbox: BoundingBox,
        frame_id: int,
        timestamp: datetime,
    ) -> LivePlateOverlay | None:
        detections = self._detect_plates(vehicle_crop, track)
        track.plate_detections_seen += len(detections)
        return self._evaluate_plate_detections(
            track,
            vehicle_crop,
            vehicle_bbox,
            detections,
            frame_id,
            timestamp,
        )

    def _evaluate_plate_detections(
        self,
        track: VehicleTrack,
        detection_image: NDArray[np.uint8],
        detection_image_bbox: BoundingBox,
        detections: list[PlateDetection],
        frame_id: int,
        timestamp: datetime,
    ) -> LivePlateOverlay | None:
        evaluated: list[
            tuple[PlateDetection, BoundingBox, NDArray[np.uint8], PlateQuality]
        ] = []
        for detection in detections:
            display_bbox = detection.bbox.clip(
                detection_image.shape[1],
                detection_image.shape[0],
            )
            if display_bbox is None:
                continue
            effective = expanded_plate_detection(
                detection,
                image_width=detection_image.shape[1],
                image_height=detection_image.shape[0],
                vehicle_type=track.vehicle_type,
                config=self._settings.vision.plate_crop,
            )
            if effective is None:
                continue
            plate_crop = self._crop(detection_image, effective.bbox)
            quality = self._quality_evaluator.evaluate(plate_crop, effective)
            if not quality.eligible:
                self._stats.quality_rejections += 1
            evaluated.append((effective, display_bbox, plate_crop, quality))
        if not evaluated:
            return None
        visual_detection, visual_bbox, _visual_crop, visual_quality = max(
            evaluated,
            key=lambda item: item[3].total_score,
        )
        prior = track.plate_observations[-1] if track.plate_observations else None
        overlay = LivePlateOverlay(
            bbox=visual_bbox.translate(detection_image_bbox.x1, detection_image_bbox.y1),
            detection_confidence=visual_detection.confidence,
            quality_score=visual_quality.total_score,
            text=prior.normalized_text if prior is not None else None,
            ocr_confidence=prior.ocr_confidence if prior is not None else None,
        )
        candidates = [item for item in evaluated if item[3].eligible]
        if not candidates:
            return overlay
        detection, clipped, plate_crop, quality = max(
            candidates,
            key=lambda item: item[3].total_score,
        )
        best_ocr: OCRResult | None = None
        best_normalization: PlateNormalization | None = None
        for variant in self._preprocessor.variants(plate_crop, quality, detection):
            self._stats.ocr_requests += 1
            try:
                inference_started = time.perf_counter()
                result = self._ocr.recognize(variant.image)
                self._stats.ocr_inference_seconds += time.perf_counter() - inference_started
            except InferenceError:
                self._stats.ocr_inference_seconds += time.perf_counter() - inference_started
                self._stats.ocr_failures += 1
                logger.exception(
                    "ocr_failed",
                    extra={"camera_id": track.camera_id, "track_id": track.logical_id},
                )
                continue
            if result.confidence < self._settings.vision.ocr.minimum_confidence:
                continue
            normalized = self._normalizer.normalize(result.text)
            if not normalized.valid:
                continue
            if best_ocr is None or result.confidence > best_ocr.confidence:
                best_ocr = result
                best_normalization = normalized
        if best_ocr is None or best_normalization is None:
            return overlay
        observation = PlateObservation(
            frame_id=frame_id,
            timestamp=timestamp,
            raw_text=best_ocr.text,
            normalized_text=best_normalization.normalized,
            compact_text=best_normalization.compact,
            ocr_confidence=best_ocr.confidence,
            detection_confidence=detection.confidence,
            quality_score=quality.total_score,
            corrections=best_normalization.corrections,
            plate_model=detection.model,
            ocr_model=best_ocr.model,
            partial=best_normalization.partial,
        )
        track.add_plate_observation(observation)
        self._stats.ocr_observations += 1
        score = self._selector.plate_score(quality, best_ocr.confidence, detection.confidence)
        if track.best_plate_crop is None or score > track.best_plate_crop.score:
            track.best_plate_crop = ImageCandidate(
                frame_id=frame_id,
                timestamp=timestamp,
                score=score,
                image=plate_crop.copy(),
            )
        return LivePlateOverlay(
            bbox=clipped.translate(detection_image_bbox.x1, detection_image_bbox.y1),
            detection_confidence=detection.confidence,
            quality_score=quality.total_score,
            text=best_normalization.normalized,
            ocr_confidence=best_ocr.confidence,
        )

    async def _report_live_preview(
        self,
        frame: VideoFrame,
        vehicles: list[LiveVehicleOverlay],
    ) -> None:
        if self._live_preview is None:
            return
        metadata = LiveFrameMetadata(
            camera_id=frame.camera_id,
            frame_id=frame.frame_id,
            stream_epoch=frame.stream_epoch,
            captured_at=frame.timestamp,
            source_width=frame.image.shape[1],
            source_height=frame.image.shape[0],
            vehicles=tuple(vehicles),
            vehicle_roi=self._roi,
            crossing_line=self._crossing_line,
        )
        try:
            await self._live_preview.report(frame.image, metadata)
        except Exception:
            logger.exception(
                "live_preview_observer_failed",
                extra={"camera_id": frame.camera_id, "frame_id": frame.frame_id},
            )

    async def _finalize_timed_out(self, now: datetime) -> list[VehicleEvent]:
        timeout = self._settings.tracking.timeout_seconds
        track_keys = [
            key
            for key, track in self._active.items()
            if (now - track.last_seen).total_seconds() >= timeout
        ]
        events: list[VehicleEvent] = []
        for key in track_keys:
            event = await self._finalize_track(key)
            if event is not None:
                events.append(event)
        return events

    async def _finalize_all(self) -> list[VehicleEvent]:
        events: list[VehicleEvent] = []
        for key in list(self._active):
            event = await self._finalize_track(key)
            if event is not None:
                events.append(event)
        return events

    async def _finalize_track(self, key: TrackKey) -> VehicleEvent | None:
        track = self._active.pop(key, None)
        if track is None:
            return None
        event = await self._finalizer.finalize(track)
        self._completed_track_ids[key] = track.last_seen
        if event is not None:
            self._stats.finalized_tracks += 1
        return event

    def _expire_completed_tracks(self, now: datetime) -> None:
        timeout = self._settings.tracking.timeout_seconds
        expired = [
            key
            for key, last_observed_at in self._completed_track_ids.items()
            if (now - last_observed_at).total_seconds() >= timeout
        ]
        for key in expired:
            del self._completed_track_ids[key]

    async def _report_health(self, *, force: bool = False) -> None:
        if self._health_reporter is None:
            return
        health = getattr(self._source, "health", None)
        if health is not None:
            elapsed = max(time.monotonic() - self._started_at, 1e-9)
            health = replace(
                health,
                sampled_frames=self._stats.sampled_frames,
                vehicle_detections=self._stats.vehicle_detections,
                plate_detections=self._stats.plate_detections,
                ocr_requests=self._stats.ocr_requests,
                ocr_success=self._stats.ocr_observations,
                events_created=self._stats.finalized_tracks,
                track_count=len(self._active),
                inference_fps=self._stats.sampled_frames / elapsed,
                vehicle_inference_latency_ms=_average_ms(
                    self._stats.vehicle_inference_seconds,
                    self._stats.vehicle_inference_calls,
                ),
                plate_inference_latency_ms=_average_ms(
                    self._stats.plate_inference_seconds,
                    self._stats.plate_inference_calls,
                ),
                ocr_latency_ms=_average_ms(
                    self._stats.ocr_inference_seconds,
                    self._stats.ocr_requests,
                ),
            )
            await self._health_reporter.report(health, force=force)

    def _inside_roi(self, bbox: BoundingBox) -> bool:
        return self._roi is None or point_in_polygon(bbox.center, self._roi)

    @classmethod
    def _pop_matching_plate(
        cls,
        detections: list[PlateDetection],
        tracked_bbox: BoundingBox,
    ) -> PlateDetection | None:
        if not detections:
            return None
        index = max(
            range(len(detections)),
            key=lambda candidate: cls._bbox_iou(
                detections[candidate].bbox,
                tracked_bbox,
            ),
        )
        if cls._bbox_iou(detections[index].bbox, tracked_bbox) <= 0:
            return None
        return detections.pop(index)

    @staticmethod
    def _bbox_iou(left: BoundingBox, right: BoundingBox) -> float:
        intersection_width = max(0, min(left.x2, right.x2) - max(left.x1, right.x1))
        intersection_height = max(0, min(left.y2, right.y2) - max(left.y1, right.y1))
        intersection = intersection_width * intersection_height
        union = left.area + right.area - intersection
        return intersection / union if union > 0 else 0.0

    @staticmethod
    def _crop(image: NDArray[np.uint8], bbox: BoundingBox) -> NDArray[np.uint8]:
        return np.ascontiguousarray(image[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2])


def _average_ms(total_seconds: float, count: int) -> float:
    return total_seconds * 1000 / count if count else 0.0
