import pytest
from minio.commonconfig import Filter
from minio.lifecycleconfig import Expiration, LifecycleConfig, Rule

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
