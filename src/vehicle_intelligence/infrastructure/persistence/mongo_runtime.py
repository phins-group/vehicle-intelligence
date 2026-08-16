"""Shared async MongoDB client and request-scoped transaction boundary."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from pymongo import AsyncMongoClient
from pymongo.errors import PyMongoError
from pymongo.read_concern import ReadConcern
from pymongo.write_concern import WriteConcern

from vehicle_intelligence.config import MongoConfig
from vehicle_intelligence.exceptions import ConfigurationError, PersistenceError


class MongoRuntime:
    def __init__(self, config: MongoConfig) -> None:
        self.config = config
        self.client = AsyncMongoClient(
            config.uri.get_secret_value(),
            tz_aware=True,
            serverSelectionTimeoutMS=config.server_selection_timeout_ms,
            connectTimeoutMS=config.connect_timeout_ms,
            socketTimeoutMS=config.socket_timeout_ms,
        )
        self.database = self.client[config.database]
        self._initialized = False
        self._transaction_active: ContextVar[bool] = ContextVar(
            f"mongo_transaction_{id(self)}",
            default=False,
        )

    async def initialize(self) -> None:
        try:
            hello = await self.client.admin.command("hello")
        except PyMongoError as exc:
            raise PersistenceError("cannot initialize shared MongoDB runtime") from exc
        if self.config.transactions_enabled:
            transactional = bool(hello.get("setName")) or hello.get("msg") == "isdbgrid"
            if not transactional or hello.get("logicalSessionTimeoutMinutes") is None:
                raise ConfigurationError(
                    "MongoDB transactions require a replica set or mongos with sessions"
                )
        self._initialized = True

    async def ping(self) -> None:
        """Verify that the shared canonical store is currently reachable."""

        if not self._initialized:
            raise PersistenceError("MongoDB runtime is not initialized")
        try:
            await self.client.admin.command("ping")
        except PyMongoError as exc:
            raise PersistenceError("MongoDB readiness probe failed") from exc

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        if not self.config.transactions_enabled:
            yield
            return
        if self._transaction_active.get():
            yield
            return
        if not self._initialized:
            raise PersistenceError("MongoDB runtime is not initialized")
        try:
            async with self.client.start_session() as session, session.bind(end_session=False):
                transaction = await session.start_transaction(
                    read_concern=ReadConcern("snapshot"),
                    write_concern=WriteConcern("majority"),
                    max_commit_time_ms=self.config.transaction_max_commit_time_ms,
                )
                token = self._transaction_active.set(True)
                try:
                    async with transaction:  # noqa: SIM117 - transaction context is awaitable
                        yield
                finally:
                    self._transaction_active.reset(token)
        except PyMongoError as exc:
            raise PersistenceError("MongoDB transaction failed") from exc

    async def close(self) -> None:
        self._initialized = False
        await self.client.close()


@dataclass(frozen=True, slots=True)
class MongoBinding:
    client: AsyncMongoClient
    database: object
    owns_client: bool


def bind_mongo(source: MongoConfig | MongoRuntime) -> MongoBinding:
    if isinstance(source, MongoRuntime):
        return MongoBinding(source.client, source.database, False)
    client = AsyncMongoClient(
        source.uri.get_secret_value(),
        tz_aware=True,
        serverSelectionTimeoutMS=source.server_selection_timeout_ms,
        connectTimeoutMS=source.connect_timeout_ms,
        socketTimeoutMS=source.socket_timeout_ms,
    )
    return MongoBinding(client, client[source.database], True)
