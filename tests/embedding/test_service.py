import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from app.embedding.service import EmbeddingService
from app.embedding.schemas import EmbeddingRequest, ClothingMetadata
from app.embedding.exceptions import ExternalAPIError, VectorDBError


# 수정
@pytest.mark.asyncio
async def test_get_embedding_success(mocker):
    mock_settings = MagicMock()
    mock_settings.upstage_api_key = "dummy_key"
    mock_settings.embedding_model = "embedding-passage"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)

    service = EmbeddingService()
    
    # Mock httpx.AsyncClient.post
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"data": [{"embedding": [0.1] * 4096}]}
    mock_response.raise_for_status = MagicMock()
    
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.return_value = mock_response

    # When
    result = await service.get_embedding("test text")

    # Then
    assert len(result) == 4096
    assert result[0] == 0.1
    mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_upsert_clothing_success(mocker, mock_get_qdrant_client):
    # Given
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
            fit="오버핏"
        )
    )
    
    # Mock get_embedding
    mocker.patch.object(service, "get_embedding", return_value=[0.1] * 4096)
    
    # Mock settings for collection name
    mock_settings = MagicMock()
    mock_settings.qdrant_collection_name = "test_collection"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)
    
    # When
    result = await service.upsert(request)

    # Then
    assert result is True
    mock_get_qdrant_client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_delete_clothing_success(mock_get_qdrant_client, mocker):
    # Given
    service = EmbeddingService()
    clothes_id = 1
    
    # Mock settings for collection name
    mock_settings = MagicMock()
    mock_settings.qdrant_collection_name = "test_collection"
    mocker.patch("app.embedding.service.get_settings", return_value=mock_settings)
    
    # When
    result = await service.delete(clothes_id)

    # Then
    assert result is True
    mock_get_qdrant_client.delete.assert_called_once()
