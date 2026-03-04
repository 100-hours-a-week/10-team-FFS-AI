from unittest.mock import AsyncMock, MagicMock

import pytest

from app.closet.segmentation import SegmentationService


@pytest.fixture
def gemini_client() -> MagicMock:
    client = MagicMock()
    client.generate_collage = AsyncMock(return_value=[b"img1", b"img2"])
    return client


@pytest.fixture
def openai_client() -> MagicMock:
    client = MagicMock()
    client.segment = AsyncMock(return_value=[b"openai_img1", b"openai_img2"])
    return client


# ── Gemini 정상 동작 ──


@pytest.mark.asyncio
async def test_segment_gemini_success(gemini_client: MagicMock) -> None:
    service = SegmentationService(gemini_client=gemini_client)
    result = await service.segment("https://example.com/image.jpg")

    assert result == [b"img1", b"img2"]
    gemini_client.generate_collage.assert_called_once()


# ── Gemini 실패 → OpenAI 폴백 ──


@pytest.mark.asyncio
async def test_segment_gemini_fails_openai_fallback(
    gemini_client: MagicMock, openai_client: MagicMock
) -> None:
    gemini_client.generate_collage = AsyncMock(side_effect=ValueError("503 error"))

    service = SegmentationService(
        gemini_client=gemini_client, openai_client=openai_client
    )
    result = await service.segment("https://example.com/image.jpg")

    assert result == [b"openai_img1", b"openai_img2"]
    openai_client.segment.assert_called_once_with("https://example.com/image.jpg")


# ── Gemini 실패 + OpenAI 없음 → 예외 전파 ──


@pytest.mark.asyncio
async def test_segment_gemini_fails_no_fallback(gemini_client: MagicMock) -> None:
    gemini_client.generate_collage = AsyncMock(side_effect=ValueError("503 error"))

    service = SegmentationService(gemini_client=gemini_client, openai_client=None)

    with pytest.raises(ValueError, match="503 error"):
        await service.segment("https://example.com/image.jpg")


# ── Gemini 실패 + OpenAI도 실패 → 예외 전파 ──


@pytest.mark.asyncio
async def test_segment_both_fail(
    gemini_client: MagicMock, openai_client: MagicMock
) -> None:
    gemini_client.generate_collage = AsyncMock(side_effect=ValueError("Gemini 503"))
    openai_client.segment = AsyncMock(side_effect=RuntimeError("OpenAI error"))

    service = SegmentationService(
        gemini_client=gemini_client, openai_client=openai_client
    )

    with pytest.raises(RuntimeError, match="OpenAI error"):
        await service.segment("https://example.com/image.jpg")
