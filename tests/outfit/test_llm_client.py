from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.outfit.exceptions import LLMError
from app.outfit.llm_client import OpenAIClient


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.openai_api_key = "test_key"
    settings.openai_chat_model = "gpt-4o-mini"
    settings.llm_timeout = 10
    return settings


@pytest.fixture
def client(mock_settings):
    return OpenAIClient(settings=mock_settings)


class TestChatCompletionSuccess:

    @pytest.mark.asyncio
    async def test_basic_completion(self, client, mocker):
        # Given
        messages = [{"role": "user", "content": "hello"}]
        expected_response = {"choices": [{"message": {"content": "world"}}]}

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = expected_response
        mock_response.raise_for_status = MagicMock()

        mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response)

        # When
        result = await client.chat_completion(messages)

        # Then
        assert result == expected_response

    @pytest.mark.asyncio
    async def test_custom_parameters(self, client, mocker):
        # Given
        messages = [{"role": "user", "content": "test"}]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response)

        # When
        await client.chat_completion(messages, temperature=0.5, max_tokens=1000)

        # Then
        payload = mock_post.call_args.kwargs["json"]
        assert payload["temperature"] == 0.5
        assert payload["max_tokens"] == 1000


class TestRetryBehavior:

    @pytest.mark.asyncio
    async def test_retry_on_timeout(self, client, mocker):
        # Given
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = [
            httpx.TimeoutException("timeout"),
            mock_response,
        ]

        # When
        result = await client.chat_completion([{"role": "user", "content": "hi"}])

        # Then
        assert result == {"choices": []}
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_429(self, client, mocker):
        # Given
        error_response = MagicMock()
        error_response.status_code = 429
        error_response.text = "Rate limit exceeded"
        error_response.headers = {"Retry-After": "1"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"choices": []}
        success_response.raise_for_status = MagicMock()

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = [
            httpx.HTTPStatusError("429", request=MagicMock(), response=error_response),
            success_response,
        ]

        # When
        result = await client.chat_completion([{"role": "user", "content": "hi"}])

        # Then
        assert result == {"choices": []}
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_on_500(self, client, mocker):
        # Given
        error_response = MagicMock()
        error_response.status_code = 500
        error_response.text = "Internal server error"
        error_response.headers = {}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"choices": []}
        success_response.raise_for_status = MagicMock()

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = [
            httpx.HTTPStatusError("500", request=MagicMock(), response=error_response),
            success_response,
        ]

        # When
        result = await client.chat_completion([{"role": "user", "content": "hi"}])

        # Then
        assert result == {"choices": []}
        assert mock_post.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, client, mocker):
        # Given
        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = httpx.TimeoutException("timeout")

        # When / Then
        with pytest.raises(LLMError, match="Network error"):
            await client.chat_completion([{"role": "user", "content": "hi"}])

        assert mock_post.call_count == 3


class TestNoRetryErrors:

    @pytest.mark.asyncio
    async def test_no_retry_on_401(self, client, mocker):
        # Given
        error_response = MagicMock()
        error_response.status_code = 401
        error_response.text = "Invalid API Key"
        error_response.headers = {}

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = httpx.HTTPStatusError(
            "401", request=MagicMock(), response=error_response
        )

        # When / Then
        with pytest.raises(LLMError, match="Invalid OpenAI API Key"):
            await client.chat_completion([{"role": "user", "content": "hi"}])

        assert mock_post.call_count == 1

    @pytest.mark.asyncio
    async def test_no_retry_on_400(self, client, mocker):
        # Given
        error_response = MagicMock()
        error_response.status_code = 400
        error_response.text = "Bad request"
        error_response.headers = {}

        mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
        mock_post.side_effect = httpx.HTTPStatusError(
            "400", request=MagicMock(), response=error_response
        )

        # When / Then
        with pytest.raises(LLMError, match="Invalid request"):
            await client.chat_completion([{"role": "user", "content": "hi"}])

        assert mock_post.call_count == 1


class TestApiKeyValidation:

    @pytest.mark.asyncio
    async def test_missing_api_key(self, mock_settings):
        # Given
        mock_settings.openai_api_key = None
        client = OpenAIClient(settings=mock_settings)

        # When / Then
        with pytest.raises(LLMError, match="OPENAI_API_KEY is not configured"):
            await client.chat_completion([{"role": "user", "content": "hi"}])
