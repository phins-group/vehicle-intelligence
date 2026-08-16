import pytest
from minio.commonconfig import Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule
from urllib3 import PoolManager, Retry, Timeout

from vehicle_intelligence.config import MinioConfig
from vehicle_intelligence.exceptions import MediaStorageError
from vehicle_intelligence.infrastructure.storage.minio import MinioMediaStorage


class FakeLifecycleClient:
    def __init__(self) -> None:
        external = Rule(
            "Enabled",
            rule_filter=Filter(prefix="external/"),
            rule_id="external-rule",
            expiration=Expiration(days=99),
        )
        self.config = LifecycleConfig([external])
        self.set_calls = 0

    def get_bucket_lifecycle(self, _bucket):
        return self.config

    def set_bucket_lifecycle(self, _bucket, config):
        self.config = config
        self.set_calls += 1


class NoLifecycleConfigurationError(Exception):
    code = "NoSuchLifecycleConfiguration"


class FakeEmptyLifecycleClient(FakeLifecycleClient):
    def __init__(self) -> None:
        self.config = None
        self.set_calls = 0

    def get_bucket_lifecycle(self, _bucket):
        if self.config is None:
            raise NoLifecycleConfigurationError
        return self.config


class FakeBucketProbeClient:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.requested_buckets: list[str] = []

    def bucket_exists(self, bucket: str) -> bool:
        self.requested_buckets.append(bucket)
        return self.exists

    def list_buckets(self) -> None:
        raise AssertionError("readiness must not require account-wide bucket listing")


class FakePutClient:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def put_object(self, *args: object, **kwargs: object) -> None:
        self.calls.append((args, kwargs))


class FakePool:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.clear_calls = 0

    def clear(self) -> None:
        self.clear_calls += 1
        if self.fail:
            raise RuntimeError("pool clear failed")


def test_minio_clients_receive_bounded_independent_network_policies() -> None:
    config = MinioConfig(
        connect_timeout_seconds=2.5,
        read_timeout_seconds=7.5,
        maximum_retries=2,
        retry_backoff_seconds=0.3,
        retry_backoff_max_seconds=1.5,
    )

    storage = MinioMediaStorage(config)
    private_pool = storage._client._http
    public_pool = storage._public_client._http

    assert isinstance(private_pool, PoolManager)
    assert isinstance(public_pool, PoolManager)
    assert private_pool is not public_pool
    assert storage._http_clients == (private_pool, public_pool)
    for pool in (private_pool, public_pool):
        timeout = pool.connection_pool_kw["timeout"]
        retries = pool.connection_pool_kw["retries"]
        assert isinstance(timeout, Timeout)
        assert timeout.total == 10
        assert timeout.connect_timeout == 2.5
        assert timeout.read_timeout == 7.5
        assert isinstance(retries, Retry)
        assert retries.total == 2
        assert retries.connect == 2
        assert retries.read == 2
        assert retries.status == 2
        assert retries.other == 0
        assert retries.redirect == 0
        assert retries.backoff_factor == 0.3
        assert retries.backoff_max == 1.5
        assert retries.status_forcelist == (500, 502, 503, 504)
        assert retries.respect_retry_after_header is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("payload", "expected_part_size"),
    (
        (b"jpeg", 5 * 1024 * 1024),
        (b"x" * (5 * 1024 * 1024 + 1), 5 * 1024 * 1024 + 1),
    ),
)
async def test_minio_put_forces_one_single_part_without_worker_fanout(
    payload: bytes,
    expected_part_size: int,
) -> None:
    storage = MinioMediaStorage(MinioConfig())
    client = FakePutClient()
    storage._client = client
    storage._bucket_ready = True

    assert await storage.put("events/one.jpg", payload, "image/jpeg") == "events/one.jpg"

    assert len(client.calls) == 1
    args, kwargs = client.calls[0]
    assert args[0:2] == ("vehicle-media", "events/one.jpg")
    assert args[3] == len(payload)
    assert kwargs["content_type"] == "image/jpeg"
    assert kwargs["part_size"] == expected_part_size
    assert kwargs["num_parallel_uploads"] == 1


@pytest.mark.asyncio
async def test_minio_put_rejects_data_above_the_single_put_limit_before_io(monkeypatch) -> None:
    from vehicle_intelligence.infrastructure.storage import minio as minio_module

    storage = MinioMediaStorage(MinioConfig())
    client = FakePutClient()
    storage._client = client
    monkeypatch.setattr(minio_module, "_MINIO_MAX_SINGLE_PUT_BYTES", 3)

    with pytest.raises(MediaStorageError, match="5 GiB single-PUT limit"):
        await storage.put("events/too-large.jpg", b"xxxx", "image/jpeg")

    assert client.calls == []


@pytest.mark.asyncio
async def test_minio_close_clears_both_pools_once_even_when_the_first_fails() -> None:
    storage = MinioMediaStorage(MinioConfig())
    first = FakePool(fail=True)
    second = FakePool()
    storage._http_clients = (first, second)

    with pytest.raises(MediaStorageError, match="cannot close MinIO HTTP clients"):
        await storage.close()
    await storage.close()

    assert first.clear_calls == 1
    assert second.clear_calls == 1


@pytest.mark.asyncio
async def test_minio_readiness_checks_only_the_configured_bucket() -> None:
    storage = MinioMediaStorage(MinioConfig(bucket="vehicle-media-probe"))
    client = FakeBucketProbeClient(exists=True)
    storage._client = client

    await storage.ping()

    assert client.requested_buckets == ["vehicle-media-probe"]


@pytest.mark.asyncio
async def test_minio_readiness_rejects_a_missing_configured_bucket() -> None:
    storage = MinioMediaStorage(MinioConfig(bucket="vehicle-media-missing"))
    client = FakeBucketProbeClient(exists=False)
    storage._client = client

    with pytest.raises(MediaStorageError, match="readiness probe failed"):
        await storage.ping()

    assert client.requested_buckets == ["vehicle-media-missing"]


@pytest.mark.asyncio
async def test_minio_lifecycle_reconciliation_preserves_external_rules_and_is_idempotent() -> None:
    storage = MinioMediaStorage(MinioConfig())
    client = FakeLifecycleClient()
    storage._client = client
    storage._bucket_ready = True

    first = await storage.reconcile_lifecycle(7)
    second = await storage.reconcile_lifecycle(7)

    assert first.changed is True
    assert first.managed_rules == 2
    assert first.preserved_rules == 1
    assert second.changed is False
    assert client.set_calls == 1
    assert {rule.rule_id for rule in client.config.rules} == {
        "external-rule",
        "vip-managed-debug-expiry",
        "vip-managed-temporary-expiry",
    }


@pytest.mark.asyncio
async def test_minio_lifecycle_initializes_a_bucket_without_existing_configuration() -> None:
    storage = MinioMediaStorage(MinioConfig())
    client = FakeEmptyLifecycleClient()
    storage._client = client
    storage._bucket_ready = True

    result = await storage.reconcile_lifecycle(7)

    assert result.changed is True
    assert result.preserved_rules == 0
    assert client.set_calls == 1
    assert len(client.config.rules) == 2


@pytest.mark.asyncio
async def test_minio_retention_rejects_unsafe_object_keys() -> None:
    storage = MinioMediaStorage(MinioConfig())
    with pytest.raises(MediaStorageError, match="unsafe"):
        await storage.remove("../outside.jpg")
