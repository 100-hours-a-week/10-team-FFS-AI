from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.gemini_client import GeminiImageAnalyzer


@pytest.fixture
def mock_genai() -> Generator[MagicMock, Any, None]:
    # gemini_client.py 안에서 `from google import genai` 를 했으므로
    # patch 타겟은 "app.closet.gemini_client.genai"
    with patch("app.closet.gemini_client.genai") as mock:
        yield mock


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    # GeminiImageAnalyzer가 get_settings()를 호출하므로 고정값 주입
    with patch("app.closet.gemini_client.get_settings") as mock:
        settings = MagicMock()
        settings.gemini_api_key = "test-api-key"
        settings.gemini_model = "gemini-2.5-flash"
        mock.return_value = settings
        yield settings


@pytest.mark.asyncio
async def test_analyze_image_success(
    mock_genai: MagicMock, mock_settings: MagicMock
) -> None:
    # Given: genai.Client() -> client_mock
    client_mock = MagicMock()
    mock_genai.Client.return_value = client_mock

    # client.aio.models.generate_content(...) -> resp
    resp = MagicMock()
    resp.text = (
        '{"major": {"category": "셔츠", "color": ["흰색"]}, '
        '"extra": {"caption": "흰색 셔츠입니다."}}'
    )

    client_mock.aio.models.generate_content = AsyncMock(return_value=resp)

    analyzer = GeminiImageAnalyzer()

    # When
    result = await analyzer.analyze_image(b"fake_image_bytes")

    # Then
    assert result["major"]["category"] == "셔츠"
    assert result["major"]["color"] == ["흰색"]
    assert result["extra"]["caption"] == "흰색 셔츠입니다."

    client_mock.aio.models.generate_content.assert_called_once()

    # 옵션까지 검증하고 싶으면(선택):
    _, kwargs = client_mock.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert "contents" in kwargs
    assert kwargs["config"].response_mime_type == "application/json"


@pytest.mark.asyncio
async def test_analyze_image_json_error(
    mock_genai: MagicMock, mock_settings: MagicMock
) -> None:
    # Given
    client_mock = MagicMock()
    mock_genai.Client.return_value = client_mock

    resp = MagicMock()
    resp.text = "Invalid JSON"
    client_mock.aio.models.generate_content = AsyncMock(return_value=resp)

    analyzer = GeminiImageAnalyzer()

    # When & Then
    with pytest.raises(ValueError, match="Invalid JSON response from Gemini"):
        await analyzer.analyze_image(b"fake_image_bytes")
