import os
import uuid
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

from vehicle_intelligence.config import MinioConfig
from vehicle_intelligence.exceptions import MediaStorageError
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage

minio = pytest.importorskip("minio")
Minio = minio.Minio


@pytest.mark.skipif(
    not os.getenv("TEST_MINIO_ENDPOINT"),
    reason="TEST_MINIO_ENDPOINT is not configured",
)
async def test_minio_media_storage_round_trip() -> None:
    endpoint = os.environ["TEST_MINIO_ENDPOINT"]
    access_key = os.getenv("TEST_MINIO_ACCESS_KEY", "minioadmin")
    secret_key = os.getenv("TEST_MINIO_SECRET_KEY", "minioadmin")
    bucket = f"vehicle-media-test-{uuid.uuid4().hex[:12]}"
    key = f"events/{uuid.uuid4().hex}/plate.jpg"
    payload = b"phase-1-minio-integration"
    config = MinioConfig(
        endpoint=endpoint,
        public_endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        bucket=bucket,
        secure=False,
    )
    storage = MinioMediaStorage(config)
    client = Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=False,
    )

    try:
        assert await storage.put(key, payload, "image/jpeg") == key
        assert await storage.exists(key)
        assert await storage.get(key, len(payload)) == payload
        with pytest.raises(MediaStorageError, match="read limit"):
            await storage.get(key, len(payload) - 1)
        signed_url = await storage.presign_get(key, timedelta(seconds=60))
        assert signed_url is not None
        parsed_url = urlparse(signed_url)
        assert parsed_url.netloc == endpoint
        signed_query = parse_qs(parsed_url.query)
        assert "X-Amz-Credential" in signed_query
        assert "X-Amz-Signature" in signed_query
        assert "X-Amz-Secret" not in signed_query
        async with httpx.AsyncClient() as http:
            signed_response = await http.get(signed_url)
        assert signed_response.status_code == 200
        assert signed_response.content == payload
        assert await storage.presign_get(
            f"events/{uuid.uuid4().hex}/missing.jpg",
            timedelta(seconds=60),
        ) is None
        assert not await storage.exists(f"events/{uuid.uuid4().hex}/missing.jpg")
        offline_public_signer = MinioMediaStorage(
            config.model_copy(update={"public_endpoint": "browser.invalid:9443"})
        )
        offline_public_url = await offline_public_signer.presign_get(
            key,
            timedelta(seconds=60),
        )
        assert offline_public_url is not None
        assert urlparse(offline_public_url).netloc == "browser.invalid:9443"
        response = client.get_object(bucket, key)
        try:
            assert response.read() == payload
        finally:
            response.close()
            response.release_conn()
    finally:
        if client.bucket_exists(bucket):
            client.remove_object(bucket, key)
            client.remove_bucket(bucket)
