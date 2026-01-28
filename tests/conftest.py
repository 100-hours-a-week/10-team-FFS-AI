from __future__ import annotations

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from app.embedding.service import EmbeddingService, get_embedding_service
from app.main import app


@pytest.fixture(autouse=True)
def mock_db_clients(mocker: MockerFixture) -> AsyncMock:
    # Qdrant Mock
    mock_qdrant: AsyncMock = AsyncMock()
    mock_qdrant.get_collections.return_value = MagicMock(collections=[])
    mocker.patch("app.core.database.AsyncQdrantClient", return_value=mock_qdrant)

    # Redis Mock
    mock_redis: AsyncMock = AsyncMock()
    mocker.patch("app.core.database.Redis", return_value=mock_redis)
    mocker.patch("app.core.database.ConnectionPool", return_value=MagicMock())

    # health check mock
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
