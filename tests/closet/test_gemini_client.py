from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.gemini_client import GeminiImageAnalyzer
from app.common.llm_schemas import (
    ImageAnalysisResult,
    ImageExtraAttributes,
    ImageExtraMetadata,
    ImageMajorAttributes,
)


@pytest.fixture
def mock_genai() -> Generator[MagicMock, Any, None]:
    with patch("app.closet.gemini_client.genai") as mock:
        yield mock


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.closet.gemini_client.get_settings") as mock:
        settings = MagicMock()
        settings.gemini_api_key = "test-api-key"
        settings.gemini_model = "gemini-2.5-flash"
        mock.return_value = settings
        yield settings


@pytest.mark.asyncio
async def test_analyze_image_success_with_parsed(
    mock_genai: MagicMock, mock_settings: MagicMock
) -> None:
    """resp.parsed가 있을 때 Pydantic 인스턴스를 직접 반환하는지 검증."""
    client_mock = MagicMock()
    mock_genai.Client.return_value = client_mock

    expected = ImageAnalysisResult(
        major=ImageMajorAttributes(
            category="TOP",
            color=["흰색"],
            material=["면"],
            style_tags=[],
        ),
        extra=ImageExtraAttributes(
            meta_data=ImageExtraMetadata(),
            caption="흰색 셔츠입니다.",
        ),
    )

    resp = MagicMock()
    resp.parsed = expected
    client_mock.aio.models.generate_content = AsyncMock(return_value=resp)

    analyzer = GeminiImageAnalyzer()
    result = await analyzer.analyze_image(b"fake_image_bytes")

    assert isinstance(result, ImageAnalysisResult)
    assert result.major.category == "TOP"
    assert result.major.color == ["흰색"]
    assert result.extra.caption == "흰색 셔츠입니다."

    _, kwargs = client_mock.aio.models.generate_content.call_args
    assert kwargs["model"] == "gemini-2.5-flash"
    assert kwargs["config"].response_schema is ImageAnalysisResult


@pytest.mark.asyncio
async def test_analyze_image_fallback_when_parsed_is_none(
    mock_genai: MagicMock, mock_settings: MagicMock
) -> None:
    """resp.parsed가 None이고 text도 없을 때 fallback을 반환하는지 검증."""
    client_mock = MagicMock()
    mock_genai.Client.return_value = client_mock

    resp = MagicMock()
    resp.parsed = None
    resp.text = None
    client_mock.aio.models.generate_content = AsyncMock(return_value=resp)

    analyzer = GeminiImageAnalyzer()
    result = await analyzer.analyze_image(b"fake_image_bytes")

    assert isinstance(result, ImageAnalysisResult)
    assert result.major.category == "ETC"


@pytest.mark.asyncio
async def test_analyze_image_raises_on_exception(
    mock_genai: MagicMock, mock_settings: MagicMock
) -> None:
    """Gemini API 에러 시 예외가 전파되는지 검증."""
    client_mock = MagicMock()
    mock_genai.Client.return_value = client_mock
    client_mock.aio.models.generate_content = AsyncMock(
        side_effect=RuntimeError("Gemini API error")
    )

    analyzer = GeminiImageAnalyzer()

    with pytest.raises(RuntimeError, match="Gemini API error"):
        await analyzer.analyze_image(b"fake_image_bytes")
