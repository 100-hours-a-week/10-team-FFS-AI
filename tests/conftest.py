from __future__ import annotations

import os
from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.embedding.service import EmbeddingService, get_embedding_service
from app.main import app


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    test_env = {
        "APP_ENV": os.getenv("APP_ENV", "ci"),
        "DEBUG": os.getenv("DEBUG", "False"),
        "QDRANT_HOST": os.getenv("QDRANT_HOST", "localhost"),
        "QDRANT_PORT": os.getenv("QDRANT_PORT", "6333"),
        "QDRANT_COLLECTION_NAME": os.getenv(
            "QDRANT_COLLECTION_NAME", "test_embeddings"
        ),
        "REDIS_HOST": os.getenv("REDIS_HOST", "localhost"),
        "REDIS_PORT": os.getenv("REDIS_PORT", "6380"),
        "REDIS_DB": os.getenv("REDIS_DB", "0"),
        "UPSTAGE_API_KEY": os.getenv("UPSTAGE_API_KEY", "test_upstage_key"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "test_openai_key"),
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "test_access_key"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "test_secret_key"),
    }

    os.environ.update(test_env)

    from app.config import get_settings

    get_settings.cache_clear()

    print("\n" + "=" * 50)
    print("🔧 Test Environment Variables")
    print("=" * 50)
    for key, value in test_env.items():
        if "KEY" in key or "SECRET" in key:
            display_value = value[:10] + "..." if len(value) > 10 else value
        else:
            display_value = value
        print(f"{key}: {display_value}")
    print("=" * 50 + "\n")

    yield

    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def mock_db_clients(mocker: MockerFixture) -> AsyncMock:
    mocker.patch("app.main.init_databases", new_callable=AsyncMock)
    mocker.patch("app.main.close_databases", new_callable=AsyncMock)
    mocker.patch("app.main.init_kafka", new_callable=AsyncMock)
    mocker.patch("app.main.close_kafka", new_callable=AsyncMock)
    mocker.patch("app.main.create_consumer", new_callable=AsyncMock)
    mocker.patch("app.main.consume_loop", new_callable=AsyncMock)
    mocker.patch(
        "app.main.check_health",
        new_callable=AsyncMock,
        return_value={"qdrant": "connected", "redis": "connected"},
    )
    mocker.patch(
        "app.main.check_kafka_health",
        new_callable=AsyncMock,
        return_value="connected",
    )

    mock_qdrant: AsyncMock = AsyncMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mocker.patch("app.core.database.AsyncQdrantClient", return_value=mock_qdrant)

    mock_redis: AsyncMock = AsyncMock()
    mocker.patch("app.core.database.Redis", return_value=mock_redis)
    mocker.patch("app.core.database.ConnectionPool", return_value=MagicMock())

    mocker.patch(
        "app.core.database.check_health",
        new_callable=AsyncMock,
        return_value={"qdrant": "connected", "redis": "connected"},
    )

    return mock_qdrant


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as c:
        yield c


@pytest.fixture
def mock_get_qdrant_client(
    mocker: MockerFixture, mock_db_clients: AsyncMock
) -> AsyncMock:
    mocker.patch(
        "app.embedding.service.get_qdrant_client", return_value=mock_db_clients
    )
    mocker.patch("app.core.database.get_qdrant_client", return_value=mock_db_clients)
    return mock_db_clients


@pytest.fixture
def mock_embedding_service() -> Generator[AsyncMock, None, None]:
    mock_service: AsyncMock = AsyncMock(spec=EmbeddingService)
    app.dependency_overrides[get_embedding_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_outfit_service() -> Generator[AsyncMock, None, None]:
    from app.outfit.service import OutfitService, get_outfit_service

    mock_service: AsyncMock = AsyncMock(spec=OutfitService)
    app.dependency_overrides[get_outfit_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_shop_service() -> Generator[AsyncMock, None, None]:
    from app.shop.service import ShopService, get_shop_service

    mock_service: AsyncMock = AsyncMock(spec=ShopService)
    app.dependency_overrides[get_shop_service] = lambda: mock_service
    yield mock_service
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def redis_client() -> Generator:
    from fakeredis import aioredis

    fake_redis = aioredis.FakeRedis(decode_responses=True)
    yield fake_redis
    await fake_redis.flushall()
    await fake_redis.aclose()
