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


@pytest.mark.asyncio
async def test_upsert_sub_category_in_embedding_text() -> None:
    """sub_category가 있으면 임베딩 텍스트 앞에 prefix로 포함된다."""
    # Given
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=[0.1] * 4096)
    mock_repository = MagicMock()
    mock_repository.upsert = AsyncMock(return_value=True)

    service = EmbeddingService(client=mock_client, repository=mock_repository)

    request = EmbeddingRequest(
        user_id=1,
        clothes_id=1,
        image_url="http://example.com/image.jpg",
        major=MajorAttributes(
            category="TOP",
            sub_category="맨투맨_스웨트",
            color=["검정"],
            material=["면"],
            style_tags=["캐주얼"],
        ),
        extra=ExtraAttributes(caption="블랙 맨투맨"),
    )

    # When
    await service.upsert(request)

    # Then — 임베딩 텍스트에 sub_category가 포함되어야 함
    call_args = mock_client.embed.call_args[0][0]
    assert "맨투맨_스웨트" in call_args
    assert "TOP" in call_args


@pytest.mark.asyncio
async def test_upsert_payload_includes_sub_category() -> None:
    """payload에 subCategory 필드가 포함된다."""
    # Given
    mock_client = MagicMock()
    mock_client.embed = AsyncMock(return_value=[0.1] * 4096)
    mock_repository = MagicMock()
    mock_repository.upsert = AsyncMock(return_value=True)

    service = EmbeddingService(client=mock_client, repository=mock_repository)

    request = EmbeddingRequest(
        user_id=1,
        clothes_id=2,
        image_url="http://example.com/image.jpg",
        major=MajorAttributes(
            category="OUTER",
            sub_category="패딩",
            color=["검정"],
            material=["폴리에스터"],
            style_tags=[],
        ),
        extra=ExtraAttributes(caption="검정 패딩"),
    )

    # When
    await service.upsert(request)

    # Then — repository.upsert에 전달된 payload에 subCategory 포함
    _, kwargs = mock_repository.upsert.call_args
    payload = kwargs["payload"]
    assert payload["subCategory"] == "패딩"
    assert payload["category"] == "OUTER"
