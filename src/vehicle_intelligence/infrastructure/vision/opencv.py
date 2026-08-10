"""OpenCV video, preprocessing, perspective, and encoding adapters."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from vehicle_intelligence.application.ports import EncodedLivePreview, ImageVariant
from vehicle_intelligence.config import PreprocessingConfig
from vehicle_intelligence.domain import PlateDetection, PlateQuality, VideoFrame
from vehicle_intelligence.exceptions import MediaStorageError, VideoSourceError


class OpenCVVideoSource:
    def __init__(
        self,
        path: str | Path,
        camera_id: str,
        fps_limit: float,
        start_time: datetime | None = None,
    ) -> None:
        self._path = Path(path).expanduser().resolve()
        if not self._path.is_file():
            raise VideoSourceError(f"video file does not exist: {self._path}")
        self._capture = cv2.VideoCapture(str(self._path))
        if not self._capture.isOpened():
            raise VideoSourceError(f"cannot open video file: {self._path}")
        source_fps = float(self._capture.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0:
            self._capture.release()
            raise VideoSourceError(f"video reports an invalid FPS: {self._path}")
        self._source_fps = source_fps
        self._sample_fps = min(fps_limit, source_fps)
        self._camera_id = camera_id
        default_start = datetime.fromtimestamp(self._path.stat().st_mtime, UTC)
        self._start_time = (start_time or default_start).astimezone(UTC)
        stat = self._path.stat()
        signature = f"{self._path}|{stat.st_size}|{stat.st_mtime_ns}".encode()
        self._source_id = f"video-{hashlib.sha256(signature).hexdigest()[:12]}"
        self.decoded_frames = 0
        self.sampled_frames = 0

    @property
    def source_id(self) -> str:
        return self._source_id

    @property
    def source_fps(self) -> float:
        return self._source_fps

    def frames(self) -> Iterator[VideoFrame]:
        next_sample_seconds = 0.0
        sample_period = 1.0 / self._sample_fps
        frame_id = 0
        while self._capture.isOpened():
            success, image = self._capture.read()
            if not success:
                break
            self.decoded_frames += 1
            media_seconds = frame_id / self._source_fps
            current_frame_id = frame_id
            frame_id += 1
            if media_seconds + (0.5 / self._source_fps) < next_sample_seconds:
                continue
            next_sample_seconds += sample_period
            self.sampled_frames += 1
            yield VideoFrame(
                camera_id=self._camera_id,
                frame_id=current_frame_id,
                timestamp=self._start_time + timedelta(seconds=media_seconds),
                image=np.ascontiguousarray(image),
            )

    def close(self) -> None:
        self._capture.release()


class AdaptivePlatePreprocessor:
    def __init__(self, config: PreprocessingConfig) -> None:
        self._config = config

    def variants(
        self,
        image: NDArray[np.uint8],
        quality: PlateQuality,
        detection: PlateDetection,
    ) -> list[ImageVariant]:
        variants = [ImageVariant("original", image)]
        if not self._config.enabled:
            return variants
        rectified = self._perspective_correct(image, detection)
        base = image if rectified is None else rectified
        if rectified is not None:
            variants.append(ImageVariant("perspective", base))
        processed = self._resize(base)
        changed = processed.shape != base.shape
        if quality.sharpness < self._config.denoise_below_sharpness:
            processed = cv2.fastNlMeansDenoisingColored(processed, None, 5, 5, 7, 21)
            changed = True
        if quality.contrast < self._config.apply_clahe_below_contrast:
            processed = self._clahe(processed)
            changed = True
        if quality.sharpness < self._config.sharpen_below_sharpness:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
            processed = cv2.filter2D(processed, -1, kernel)
            changed = True
        if changed:
            variants.append(ImageVariant("adaptive", np.ascontiguousarray(processed)))
        return variants

    def _resize(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        width = self._config.resize_width
        if image.shape[1] >= width:
            return image
        height = max(1, round(image.shape[0] * width / image.shape[1]))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

    def _clahe(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self._config.clahe_clip_limit, tileGridSize=(8, 8))
        enhanced = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge((enhanced, channel_a, channel_b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _perspective_correct(
        image: NDArray[np.uint8], detection: PlateDetection
    ) -> NDArray[np.uint8] | None:
        if detection.corners is None:
            return None
        points = np.asarray(
            [
                [point.x - detection.bbox.x1, point.y - detection.bbox.y1]
                for point in detection.corners
            ],
            dtype=np.float32,
        )
        top_left, top_right, bottom_right, bottom_left = points
        width = int(
            max(np.linalg.norm(top_right - top_left), np.linalg.norm(bottom_right - bottom_left))
        )
        height = int(
            max(np.linalg.norm(bottom_left - top_left), np.linalg.norm(bottom_right - top_right))
        )
        if width < 2 or height < 2:
            return None
        destination = np.asarray(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(points, destination)
        return cv2.warpPerspective(image, transform, (width, height))


class OpenCVImageEncoder:
    def __init__(self, jpeg_quality: int = 92) -> None:
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        self._jpeg_quality = jpeg_quality

    def encode_jpeg(self, image: NDArray[np.uint8]) -> bytes:
        success, buffer = cv2.imencode(
            ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        )
        if not success:
            raise MediaStorageError("OpenCV could not encode JPEG media")
        return buffer.tobytes()


class OpenCVLivePreviewEncoder:
    def encode(
        self,
        image: NDArray[np.uint8],
        maximum_width: int,
        jpeg_quality: int,
    ) -> EncodedLivePreview:
        if maximum_width <= 0 or not 1 <= jpeg_quality <= 100:
            raise ValueError("live preview encoding configuration is invalid")
        preview = image
        if image.shape[1] > maximum_width:
            height = max(1, round(image.shape[0] * maximum_width / image.shape[1]))
            preview = cv2.resize(image, (maximum_width, height), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode(
            ".jpg",
            preview,
            [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
        )
        if not success:
            raise MediaStorageError("OpenCV could not encode live preview JPEG")
        return EncodedLivePreview(
            jpeg=buffer.tobytes(),
            width=int(preview.shape[1]),
            height=int(preview.shape[0]),
        )
