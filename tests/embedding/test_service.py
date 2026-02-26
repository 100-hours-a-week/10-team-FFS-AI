from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.closet.schemas import (
    EmbeddingRequest,
    ExtraAttributes,
    ExtraMetadata,
    MajorAttributes,
)
from app.embedding.service import EmbeddingService


@pytest.mark.asyncio
async def test_upsert_clothing_success() -> None:
    # Given
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=[0.1] * 4096)

    mock_repository = MagicMock()
    mock_repository.upsert = AsyncMock(return_value=True)

    service = EmbeddingService(
        client=mock_client,
        repository=mock_repository,
    )

    request = EmbeddingRequest(
        user_id=123,
        clothes_id=1,
        image_url="http://example.com/image.jpg",
        major=MajorAttributes(
            category="TOP",
            color=["빨강"],
            material=["니트"],
            style_tags=["캐주얼"],
        ),
        extra=ExtraAttributes(
            meta_data=ExtraMetadata(
                gender="남성",
                season=["겨울"],
                formality="캐주얼",
                fit="오버핏",
            ),
            caption="test caption",
        ),
    )

    # When
    result = await service.upsert(request)

    # Then
    assert result is True
    mock_client.embed.assert_called_once()
    mock_repository.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_delete_clothing_success() -> None:
    # Given
    mock_repository = MagicMock()
    mock_repository.delete = AsyncMock(return_value=True)

    service = EmbeddingService(repository=mock_repository)

    # When
    result = await service.delete(clothes_id=1)

    # Then
    assert result is True
    mock_repository.delete.assert_called_once_with(point_id=1)


@pytest.mark.asyncio
async def test_upsert_formats_text_correctly() -> None:
    # Given
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=[0.1] * 4096)

    mock_repository = MagicMock()
    mock_repository.upsert = AsyncMock(return_value=True)

    service = EmbeddingService(
        client=mock_client,
        repository=mock_repository,
    )

    request = EmbeddingRequest(
        user_id=123,
        clothes_id=1,
        image_url="http://example.com/image.jpg",
        major=MajorAttributes(
            category="TOP",
            color=["검정"],
            material=["울"],
            style_tags=["미니멀"],
        ),
        extra=ExtraAttributes(
            meta_data=ExtraMetadata(
                gender="여성",
                season=["겨울"],
                formality="포멀",
            ),
            caption="오버핏 더블 코트",
        ),
    )

    # When
    await service.upsert(request)

    # Then
    call_args = mock_client.embed.call_args[0][0]
    assert "검정" in call_args
    assert "울" in call_args
    assert "코트" in call_args
