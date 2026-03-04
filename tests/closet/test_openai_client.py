import base64
import json
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.openai_client import OpenAISegmentationClient


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.closet.openai_client.get_settings") as mock:
        settings = MagicMock()
        settings.openai_api_key = "test-key"
        settings.llm_timeout = 30
        settings.llm_max_retries = 3
        mock.return_value = settings
        yield settings


@pytest.fixture
def mock_openai(mock_settings: MagicMock) -> Generator[MagicMock, Any, None]:
    with patch("app.closet.openai_client.openai.AsyncOpenAI") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        yield mock_client


# ── 초기화 테스트 ──


def test_init_requires_api_key() -> None:
    with patch("app.closet.openai_client.get_settings") as mock:
        settings = MagicMock()
        settings.openai_api_key = None
        mock.return_value = settings

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAISegmentationClient()


def test_init_success(mock_settings: MagicMock) -> None:
    with patch("app.closet.openai_client.openai.AsyncOpenAI"):
        client = OpenAISegmentationClient()
        assert client is not None


# ── _identify_items 테스트 ──


@pytest.mark.asyncio
async def test_identify_items_success(mock_openai: MagicMock) -> None:
    items = ["흰색 티셔츠", "청바지", "스니커즈"]
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = json.dumps(items)
    mock_openai.chat.completions.create = AsyncMock(return_value=resp)

    client = OpenAISegmentationClient()
    result = await client._identify_items(b"fake_image")

    assert result == items


@pytest.mark.asyncio
async def test_identify_items_with_extra_text(mock_openai: MagicMock) -> None:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = '결과: ["티셔츠", "바지"]'
    mock_openai.chat.completions.create = AsyncMock(return_value=resp)

    client = OpenAISegmentationClient()
    result = await client._identify_items(b"fake_image")

    assert result == ["티셔츠", "바지"]


@pytest.mark.asyncio
async def test_identify_items_invalid_json(mock_openai: MagicMock) -> None:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "아이템을 찾을 수 없습니다"
    mock_openai.chat.completions.create = AsyncMock(return_value=resp)

    client = OpenAISegmentationClient()
    result = await client._identify_items(b"fake_image")

    assert result == []


# ── _isolate_item 테스트 ──


@pytest.mark.asyncio
async def test_isolate_item_success(mock_openai: MagicMock) -> None:
    fake_image = base64.b64encode(b"fake_generated_image").decode()
    result_data = MagicMock()
    result_data.b64_json = fake_image
    resp = MagicMock()
    resp.data = [result_data]
    mock_openai.images.edit = AsyncMock(return_value=resp)

    client = OpenAISegmentationClient()
    result = await client._isolate_item(b"input_image", "흰색 티셔츠")

    assert result == b"fake_generated_image"


@pytest.mark.asyncio
async def test_isolate_item_no_data(mock_openai: MagicMock) -> None:
    resp = MagicMock()
    resp.data = []
    mock_openai.images.edit = AsyncMock(return_value=resp)

    client = OpenAISegmentationClient()
    result = await client._isolate_item(b"input_image", "티셔츠")

    assert result is None


# ── segment 통합 테스트 ──


@pytest.mark.asyncio
async def test_segment_success(mock_openai: MagicMock) -> None:
    # _download_image mock
    with patch("app.closet.openai_client.httpx.AsyncClient") as mock_http:
        http_client = AsyncMock()
        http_resp = MagicMock()
        http_resp.content = b"downloaded_image"
        http_resp.raise_for_status = MagicMock()
        http_client.get = AsyncMock(return_value=http_resp)
        http_client.__aenter__ = AsyncMock(return_value=http_client)
        http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http.return_value = http_client

        # _identify_items mock
        identify_resp = MagicMock()
        identify_resp.choices = [MagicMock()]
        identify_resp.choices[0].message.content = '["티셔츠", "바지"]'
        mock_openai.chat.completions.create = AsyncMock(return_value=identify_resp)

        # _isolate_item mock
        fake_b64 = base64.b64encode(b"generated").decode()
        item_data = MagicMock()
        item_data.b64_json = fake_b64
        edit_resp = MagicMock()
        edit_resp.data = [item_data]
        mock_openai.images.edit = AsyncMock(return_value=edit_resp)

        client = OpenAISegmentationClient()
        results = await client.segment("https://example.com/image.jpg")

    assert len(results) == 2
    assert all(r == b"generated" for r in results)


@pytest.mark.asyncio
async def test_segment_no_items_identified(mock_openai: MagicMock) -> None:
    with patch("app.closet.openai_client.httpx.AsyncClient") as mock_http:
        http_client = AsyncMock()
        http_resp = MagicMock()
        http_resp.content = b"downloaded_image"
        http_resp.raise_for_status = MagicMock()
        http_client.get = AsyncMock(return_value=http_resp)
        http_client.__aenter__ = AsyncMock(return_value=http_client)
        http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http.return_value = http_client

        identify_resp = MagicMock()
        identify_resp.choices = [MagicMock()]
        identify_resp.choices[0].message.content = "없음"
        mock_openai.chat.completions.create = AsyncMock(return_value=identify_resp)

        client = OpenAISegmentationClient()
        with pytest.raises(ValueError, match="식별하지 못했습니다"):
            await client.segment("https://example.com/image.jpg")
