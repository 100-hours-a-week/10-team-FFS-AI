from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.gemini_client import GeminiImageAnalyzer


@pytest.fixture
def mock_genai() -> Generator[MagicMock | AsyncMock, Any, None]:
    with patch("app.closet.gemini_client.genai") as mock:
        yield mock


@pytest.mark.asyncio
async def test_analyze_image_success(mock_genai: MagicMock) -> None:
    # Given
    mock_model = MagicMock()
    mock_response = MagicMock()

    mock_response.text = '{"major": {"category": "셔츠", "color": ["흰색"]}, "extra": {"caption": "흰색 셔츠입니다."}}'

    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    mock_genai.GenerativeModel.return_value = mock_model

    analyzer = GeminiImageAnalyzer()

    # When
    result = await analyzer.analyze_image(b"fake_image_bytes")

    # Then
    assert result["major"]["category"] == "셔츠"
    assert result["major"]["color"] == ["흰색"]
    assert result["extra"]["caption"] == "흰색 셔츠입니다."
    mock_model.generate_content_async.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_image_json_error(mock_genai: MagicMock) -> None:
    # Given
    mock_model = MagicMock()
    mock_response = MagicMock()

    mock_response.text = "Invalid JSON"

    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    mock_genai.GenerativeModel.return_value = mock_model

    analyzer = GeminiImageAnalyzer()

    # When & Then
    with pytest.raises(ValueError, match="Invalid JSON"):
        await analyzer.analyze_image(b"fake_image_bytes")
