from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pytest_mock import MockerFixture

from app.embedding.schemas import ClothingMetadata, EmbeddingRequest
from app.embedding.service import EmbeddingService


@pytest.mark.asyncio
async def test_get_embedding_success(mocker: MockerFixture) -> None:
    # Given: settings must be patched BEFORE service instantiation
    mock_settings = MagicMock()
    mock_settings.upstage_api_key = "dummy_key"
    mock_settings.embedding_model = "embedding-passage"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)

    service = EmbeddingService()

    # Mock httpx.AsyncClient.post response
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 4096}]}
    mock_response.raise_for_status = MagicMock()

    mock_post: AsyncMock = mocker.patch(
        "httpx.AsyncClient.post", new_callable=AsyncMock
    )
    mock_post.return_value = mock_response

    # When
    result = await service.get_embedding("test text")

    # Then
    assert len(result) == 4096
    assert result[0] == 0.1
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_clothing_success(
    mocker: MockerFixture,
    mock_get_qdrant_client: AsyncMock,
) -> None:
    # Given: settings patched BEFORE service instantiation
    mock_settings = MagicMock()
    mock_settings.upstage_api_key = "dummy_key"
    mock_settings.embedding_model = "embedding-passage"
    mock_settings.qdrant_collection_name = "test_collection"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)

    service = EmbeddingService()
    request = EmbeddingRequest(
        user_id="user123",
        clothes_id=1,
        image_url="http://example.com/image.jpg",
        caption="test caption",
        metadata=ClothingMetadata(
            category="상의",
            color=["빨강"],
            material=["니트"],
            style_tags=["캐주얼"],
            gender="남성",
            season=["겨울"],
            formality="캐주얼",
            fit="오버핏",
        ),
    )

    # Mock get_embedding (async method)
    mocker.patch.object(
        service, "get_embedding", new_callable=AsyncMock, return_value=[0.1] * 4096
    )

    # When
    result = await service.upsert(request)

    # Then
    assert result is True
    mock_get_qdrant_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_delete_clothing_success(
    mocker: MockerFixture,
    mock_get_qdrant_client: AsyncMock,
) -> None:
    # Given: settings patched BEFORE service instantiation
    mock_settings = MagicMock()
    mock_settings.qdrant_collection_name = "test_collection"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)

    service = EmbeddingService()
    clothes_id = 1

    # When
    result = await service.delete(clothes_id)

    # Then
    assert result is True
    mock_get_qdrant_client.delete.assert_called_once()
