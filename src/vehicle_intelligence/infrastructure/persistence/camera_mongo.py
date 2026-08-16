"""MongoDB camera configuration and latest-health repositories."""

from __future__ import annotations

from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING, IndexModel, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

from vehicle_intelligence.application.ports import (
    CameraCreateOutcome,
    CredentialCipher,
    EncryptedCameraCredential,
)
from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.domain import (
    Camera,
    CameraDirection,
    CameraHealth,
    CameraStatus,
    Direction,
    Point,
    SecretUri,
)
from vehicle_intelligence.exceptions import PersistenceError
from vehicle_intelligence.infrastructure.persistence.constants import (
    CAMERA_HEALTH,
    CAMERAS,
    SYSTEM_CONFIG,
)
from vehicle_intelligence.infrastructure.persistence.mongo_runtime import MongoRuntime, bind_mongo


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class MongoCameraRepository:
    def __init__(self, config: MongoConfig | MongoRuntime, cipher: CredentialCipher) -> None:
        self._cipher = cipher
        self._runtime = config if isinstance(config, MongoRuntime) else None
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[CAMERAS]
        self._capacity = binding.database[SYSTEM_CONFIG]
        self._capacity_id = "camera-capacity"

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_indexes(
                [
                    IndexModel(
                        [("enabled", ASCENDING), ("name", ASCENDING), ("_id", ASCENDING)],
                        name="ix_enabled_name",
                    ),
                    IndexModel([("updatedAt", DESCENDING)], name="ix_updated_at"),
                ]
            )
            camera_count = await self._collection.count_documents({})
            # Another repository can initialize the shared counter concurrently.
            with suppress(DuplicateKeyError):
                await self._capacity.update_one(
                    {"_id": self._capacity_id},
                    {"$setOnInsert": {"reservedCount": camera_count}},
                    upsert=True,
                )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB camera indexes") from exc

    async def create(self, camera: Camera) -> bool:
        if self._runtime is None:
            outcome = await self._create(camera, maximum_cameras=None)
        else:
            async with self._runtime.transaction():
                outcome = await self._create(camera, maximum_cameras=None)
        return outcome is CameraCreateOutcome.CREATED

    async def create_with_capacity(
        self,
        camera: Camera,
        maximum_cameras: int,
    ) -> CameraCreateOutcome:
        if maximum_cameras < 1:
            raise ValueError("camera capacity must be positive")
        if self._runtime is None:
            return await self._create(camera, maximum_cameras=maximum_cameras)
        async with self._runtime.transaction():
            return await self._create(camera, maximum_cameras=maximum_cameras)

    async def _create(
        self,
        camera: Camera,
        maximum_cameras: int | None,
    ) -> CameraCreateOutcome:
        try:
            if await self._collection.find_one({"_id": camera.id}, {"_id": 1}) is not None:
                return CameraCreateOutcome.CONFLICT
            if not await self._reserve_capacity(maximum_cameras):
                if await self._collection.find_one({"_id": camera.id}, {"_id": 1}) is not None:
                    return CameraCreateOutcome.CONFLICT
                return CameraCreateOutcome.CAPACITY_REACHED
            try:
                await self._collection.insert_one(self._to_document(camera))
            except DuplicateKeyError:
                await self._release_capacity()
                return CameraCreateOutcome.CONFLICT
            except PyMongoError:
                await self._release_capacity()
                raise
            return CameraCreateOutcome.CREATED
        except PyMongoError as exc:
            raise PersistenceError(f"cannot create camera: {camera.id}") from exc

    async def _reserve_capacity(self, maximum_cameras: int | None) -> bool:
        if maximum_cameras is None:
            await self._capacity.update_one(
                {"_id": self._capacity_id},
                {"$inc": {"reservedCount": 1}},
                upsert=True,
            )
            return True
        reservation = await self._capacity.find_one_and_update(
            {"_id": self._capacity_id, "reservedCount": {"$lt": maximum_cameras}},
            {"$inc": {"reservedCount": 1}},
            return_document=ReturnDocument.AFTER,
        )
        return reservation is not None

    async def _release_capacity(self) -> None:
        await self._capacity.update_one(
            {"_id": self._capacity_id, "reservedCount": {"$gt": 0}},
            {"$inc": {"reservedCount": -1}},
        )

    async def replace(self, camera: Camera, expected_revision: int) -> bool:
        if camera.revision != expected_revision + 1:
            raise ValueError("replacement camera revision must increment by one")
        try:
            result = await self._collection.replace_one(
                {"_id": camera.id, "revision": expected_revision},
                self._to_document(camera),
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot update camera: {camera.id}") from exc

    async def get(self, camera_id: str) -> Camera | None:
        try:
            document = await self._collection.find_one({"_id": camera_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read camera: {camera_id}") from exc
        return self._from_document(document) if document is not None else None

    async def list(self, enabled_only: bool = False) -> list[Camera]:
        query = {"enabled": True} if enabled_only else {}
        try:
            documents = [
                document
                async for document in self._collection.find(query).sort(
                    [("name", ASCENDING), ("_id", ASCENDING)]
                )
            ]
        except PyMongoError as exc:
            raise PersistenceError("cannot list cameras") from exc
        return [self._from_document(document) for document in documents]

    async def count(self) -> int:
        try:
            return await self._collection.count_documents({})
        except PyMongoError as exc:
            raise PersistenceError("cannot count cameras") from exc

    async def list_encrypted_credentials(
        self,
        after_camera_id: str | None,
        limit: int,
    ) -> tuple[EncryptedCameraCredential, ...]:
        query = {"_id": {"$gt": after_camera_id}} if after_camera_id is not None else {}
        try:
            cursor = (
                self._collection.find(query, {"stream.rtspUrlEncrypted": 1})
                .sort([("_id", ASCENDING)])
                .limit(limit)
            )
            values = [
                EncryptedCameraCredential(
                    camera_id=str(document["_id"]),
                    token=str(document["stream"]["rtspUrlEncrypted"]),
                )
                async for document in cursor
            ]
            return tuple(values)
        except PyMongoError as exc:
            raise PersistenceError("cannot list encrypted camera credentials") from exc

    async def replace_encrypted_credential(
        self,
        camera_id: str,
        expected_token: str,
        replacement_token: str,
        rotated_at: datetime,
    ) -> bool:
        try:
            result = await self._collection.update_one(
                {
                    "_id": camera_id,
                    "stream.rtspUrlEncrypted": expected_token,
                },
                {
                    "$set": {
                        "stream.rtspUrlEncrypted": replacement_token,
                        "credentialRotatedAt": rotated_at.astimezone(UTC),
                    }
                },
            )
            return result.matched_count == 1
        except PyMongoError as exc:
            raise PersistenceError(f"cannot rotate camera credential: {camera_id}") from exc

    async def delete(self, camera_id: str) -> bool:
        if self._runtime is None:
            return await self._delete(camera_id)
        async with self._runtime.transaction():
            return await self._delete(camera_id)

    async def _delete(self, camera_id: str) -> bool:
        try:
            result = await self._collection.delete_one({"_id": camera_id})
            if result.deleted_count != 1:
                return False
            await self._release_capacity()
            return True
        except PyMongoError as exc:
            raise PersistenceError(f"cannot delete camera: {camera_id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    def _to_document(self, camera: Camera) -> dict[str, Any]:
        return {
            "_id": camera.id,
            "schemaVersion": camera.schema_version,
            "revision": camera.revision,
            "name": camera.name,
            "stream": {
                "rtspUrlEncrypted": self._cipher.encrypt(camera.rtsp_url.reveal(), camera.id),
                "fpsLimit": camera.fps_limit,
            },
            "location": {"name": camera.location, "zone": camera.zone},
            "direction": camera.direction.value,
            "vision": {
                "vehicleConfidence": camera.vehicle_confidence,
                "plateConfidence": camera.plate_confidence,
            },
            "geometry": {
                "vehicleRoi": (
                    [[point.x, point.y] for point in camera.roi] if camera.roi is not None else None
                ),
                "crossingLine": (
                    [[point.x, point.y] for point in camera.crossing_line]
                    if camera.crossing_line is not None
                    else None
                ),
                "crossingPositiveToNegative": camera.crossing_positive_to_negative.value,
                "finalizeOnCrossing": camera.finalize_on_crossing,
            },
            "enabled": camera.enabled,
            "metadata": camera.metadata,
            "createdAt": camera.created_at.astimezone(UTC),
            "updatedAt": camera.updated_at.astimezone(UTC),
        }

    def _from_document(self, document: dict[str, Any]) -> Camera:
        stream = document["stream"]
        location = document.get("location") or {}
        vision = document.get("vision") or {}
        geometry = document.get("geometry") or {}
        roi = geometry.get("vehicleRoi")
        line = geometry.get("crossingLine")
        return Camera(
            id=str(document["_id"]),
            name=str(document["name"]),
            rtsp_url=SecretUri(
                self._cipher.decrypt(str(stream["rtspUrlEncrypted"]), str(document["_id"]))
            ),
            fps_limit=float(stream["fpsLimit"]),
            direction=CameraDirection(document["direction"]),
            enabled=bool(document["enabled"]),
            vehicle_confidence=float(vision["vehicleConfidence"]),
            plate_confidence=float(vision["plateConfidence"]),
            location=location.get("name"),
            zone=location.get("zone"),
            roi=tuple(Point(float(x), float(y)) for x, y in roi) if roi else None,
            crossing_line=(tuple(Point(float(x), float(y)) for x, y in line) if line else None),
            crossing_positive_to_negative=Direction(
                geometry.get("crossingPositiveToNegative", "ENTER")
            ),
            finalize_on_crossing=bool(geometry.get("finalizeOnCrossing", False)),
            metadata=document.get("metadata") or {},
            schema_version=int(document.get("schemaVersion", 1)),
            revision=int(document["revision"]),
            created_at=_aware(document["createdAt"]),
            updated_at=_aware(document["updatedAt"]),
        )


class MongoCameraHealthRepository:
    def __init__(self, config: MongoConfig | MongoRuntime) -> None:
        binding = bind_mongo(config)
        self._client = binding.client
        self._owns_client = binding.owns_client
        self._collection = binding.database[CAMERA_HEALTH]

    async def ensure_indexes(self) -> None:
        try:
            await self._client.admin.command("ping")
            await self._collection.create_index(
                [("status", ASCENDING), ("updatedAt", DESCENDING)],
                name="ix_health_status_updated",
            )
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize MongoDB camera-health indexes") from exc

    async def save(self, health: CameraHealth) -> None:
        try:
            await self._collection.replace_one(
                {"_id": health.camera_id},
                self._to_document(health),
                upsert=True,
            )
        except PyMongoError as exc:
            raise PersistenceError(f"cannot persist camera health: {health.camera_id}") from exc

    async def get(self, camera_id: str) -> CameraHealth | None:
        try:
            document = await self._collection.find_one({"_id": camera_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot read camera health: {camera_id}") from exc
        return self._from_document(document) if document is not None else None

    async def list(self) -> list[CameraHealth]:
        try:
            documents = [
                document async for document in self._collection.find().sort([("_id", ASCENDING)])
            ]
        except PyMongoError as exc:
            raise PersistenceError("cannot list camera health") from exc
        return [self._from_document(document) for document in documents]

    async def delete(self, camera_id: str) -> None:
        try:
            await self._collection.delete_one({"_id": camera_id})
        except PyMongoError as exc:
            raise PersistenceError(f"cannot delete camera health: {camera_id}") from exc

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    @staticmethod
    def _to_document(health: CameraHealth) -> dict[str, Any]:
        return {
            "_id": health.camera_id,
            "schemaVersion": 1,
            "status": health.status.value,
            "fps": {"source": health.source_fps, "decode": health.decode_fps},
            "queueSize": health.queue_size,
            "droppedFrames": health.dropped_frames,
            "reconnectCount": health.reconnect_count,
            "connectionFailures": health.connection_failures,
            "streamEpoch": health.stream_epoch,
            "lastFrameAt": health.last_frame_at,
            "vision": {
                "decodedFrames": health.decoded_frames,
                "sampledFrames": health.sampled_frames,
                "vehicleDetections": health.vehicle_detections,
                "plateDetections": health.plate_detections,
                "ocrRequests": health.ocr_requests,
                "ocrSuccess": health.ocr_success,
                "eventsCreated": health.events_created,
                "trackCount": health.track_count,
                "inferenceFps": health.inference_fps,
                "latencyMs": {
                    "vehicle": health.vehicle_inference_latency_ms,
                    "plate": health.plate_inference_latency_ms,
                    "ocr": health.ocr_latency_ms,
                },
            },
            "updatedAt": health.updated_at.astimezone(UTC),
        }

    @staticmethod
    def _from_document(document: dict[str, Any]) -> CameraHealth:
        fps = document.get("fps") or {}
        vision = document.get("vision") or {}
        latency = vision.get("latencyMs") or {}
        return CameraHealth(
            camera_id=str(document["_id"]),
            status=CameraStatus(document["status"]),
            source_fps=float(fps.get("source", 0)),
            decode_fps=float(fps.get("decode", 0)),
            queue_size=int(document.get("queueSize", 0)),
            dropped_frames=int(document.get("droppedFrames", 0)),
            reconnect_count=int(document.get("reconnectCount", 0)),
            connection_failures=int(document.get("connectionFailures", 0)),
            stream_epoch=int(document.get("streamEpoch", 0)),
            last_frame_at=(
                _aware(document["lastFrameAt"]) if document.get("lastFrameAt") else None
            ),
            updated_at=_aware(document["updatedAt"]),
            decoded_frames=int(vision.get("decodedFrames", 0)),
            sampled_frames=int(vision.get("sampledFrames", 0)),
            vehicle_detections=int(vision.get("vehicleDetections", 0)),
            plate_detections=int(vision.get("plateDetections", 0)),
            ocr_requests=int(vision.get("ocrRequests", 0)),
            ocr_success=int(vision.get("ocrSuccess", 0)),
            events_created=int(vision.get("eventsCreated", 0)),
            track_count=int(vision.get("trackCount", 0)),
            inference_fps=float(vision.get("inferenceFps", 0)),
            vehicle_inference_latency_ms=float(latency.get("vehicle", 0)),
            plate_inference_latency_ms=float(latency.get("plate", 0)),
            ocr_latency_ms=float(latency.get("ocr", 0)),
        )
