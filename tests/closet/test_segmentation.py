from unittest.mock import AsyncMock, MagicMock

import pytest

from app.closet.segmentation import SegmentationService


@pytest.fixture
def gemini_client() -> MagicMock:
    client = MagicMock()
    client.generate_collage = AsyncMock(return_value=[b"img1", b"img2"])
    return client


# ── Gemini 정상 동작 ──


@pytest.mark.asyncio
async def test_segment_gemini_success(gemini_client: MagicMock) -> None:
    service = SegmentationService(gemini_client=gemini_client)
    result = await service.segment("https://example.com/image.jpg")

    assert result == [b"img1", b"img2"]
    gemini_client.generate_collage.assert_called_once()


# ── Gemini 실패 → 예외 전파 ──


@pytest.mark.asyncio
async def test_segment_gemini_fails_raises(gemini_client: MagicMock) -> None:
    gemini_client.generate_collage = AsyncMock(side_effect=ValueError("503 error"))

    service = SegmentationService(gemini_client=gemini_client)

    with pytest.raises(ValueError, match="503 error"):
        await service.segment("https://example.com/image.jpg")
