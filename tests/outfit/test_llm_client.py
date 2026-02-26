from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from pydantic import BaseModel

from app.outfit.exceptions import LLMError
from app.outfit.llm_client import OpenAIClient


@pytest.fixture
def mock_settings() -> MagicMock:
    settings = MagicMock()
    settings.openai_api_key = "test_key"
    settings.openai_chat_model = "gpt-4o-mini"
    settings.openai_base_url = None
    settings.llm_timeout = 10
    settings.llm_max_retries = 0  # 테스트에서 재시도 비활성화
    return settings


@pytest.fixture
def client(mock_settings: MagicMock) -> OpenAIClient:
    return OpenAIClient(settings=mock_settings)


def _make_create_response(content: str = "hello") -> MagicMock:
    """chat.completions.create 응답 mock 생성."""
    choice = MagicMock()
    choice.message.content = content
    completion = MagicMock()
    completion.choices = [choice]
    completion.model_dump.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return completion


def _make_parse_response(parsed: BaseModel | None) -> MagicMock:
    """beta.chat.completions.parse 응답 mock 생성."""
    choice = MagicMock()
    choice.message.parsed = parsed
    completion = MagicMock()
    completion.choices = [choice]
    return completion


class TestChatCompletionWithoutStructuredOutput:
    """response_format=None 일반 호출 테스트."""

    @pytest.mark.asyncio
    async def test_basic_completion(
        self: "TestChatCompletionWithoutStructuredOutput",
        client: OpenAIClient,
    ) -> None:
        messages = [{"role": "user", "content": "hello"}]
        mock_completion = _make_create_response("world")

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            result = await client.chat_completion(messages)

        assert result == {"choices": [{"message": {"content": "world"}}]}

    @pytest.mark.asyncio
    async def test_custom_parameters(
        self: "TestChatCompletionWithoutStructuredOutput",
        client: OpenAIClient,
    ) -> None:
        messages = [{"role": "user", "content": "test"}]
        mock_completion = _make_create_response()

        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ) as mock_create:
            await client.chat_completion(messages, temperature=0.5, max_tokens=1000)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 1000


class TestChatCompletionWithStructuredOutput:
    """response_format 전달 시 Structured Output 경로 테스트."""

    @pytest.mark.asyncio
    async def test_returns_parsed_pydantic_instance(
        self: "TestChatCompletionWithStructuredOutput",
        client: OpenAIClient,
    ) -> None:
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            value: str

        expected = DummyModel(value="test_value")
        mock_completion = _make_parse_response(expected)

        with patch.object(
            client._client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            result = await client.chat_completion(
                [{"role": "user", "content": "hi"}],
                response_format=DummyModel,
            )

        assert isinstance(result, DummyModel)
        assert result.value == "test_value"

    @pytest.mark.asyncio
    async def test_raises_when_parsed_is_none(
        self: "TestChatCompletionWithStructuredOutput",
        client: OpenAIClient,
    ) -> None:
        from pydantic import BaseModel

        class DummyModel(BaseModel):
            value: str

        mock_completion = _make_parse_response(None)

        with patch.object(
            client._client.beta.chat.completions,
            "parse",
            new_callable=AsyncMock,
            return_value=mock_completion,
        ):
            with pytest.raises(LLMError, match="parsing returned None"):
                await client.chat_completion(
                    [{"role": "user", "content": "hi"}],
                    response_format=DummyModel,
                )


class TestErrorHandling:
    """에러 핸들링 테스트."""

    @pytest.mark.asyncio
    async def test_authentication_error(
        self: "TestErrorHandling",
        client: OpenAIClient,
    ) -> None:
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=openai.AuthenticationError(
                "invalid key", response=MagicMock(), body={}
            ),
        ):
            with pytest.raises(LLMError, match="Invalid OpenAI API Key"):
                await client.chat_completion([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_bad_request_error(
        self: "TestErrorHandling",
        client: OpenAIClient,
    ) -> None:
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=openai.BadRequestError(
                "bad request", response=MagicMock(), body={}
            ),
        ):
            with pytest.raises(LLMError, match="Invalid request"):
                await client.chat_completion([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_rate_limit_error(
        self: "TestErrorHandling",
        client: OpenAIClient,
    ) -> None:
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=openai.RateLimitError(
                "rate limit", response=MagicMock(), body={}
            ),
        ):
            with pytest.raises(LLMError, match="failed after retries"):
                await client.chat_completion([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_connection_error(
        self: "TestErrorHandling",
        client: OpenAIClient,
    ) -> None:
        with patch.object(
            client._client.chat.completions,
            "create",
            new_callable=AsyncMock,
            side_effect=openai.APIConnectionError(request=MagicMock()),
        ):
            with pytest.raises(LLMError, match="Network error"):
                await client.chat_completion([{"role": "user", "content": "hi"}])


class TestApiKeyValidation:
    @pytest.mark.asyncio
    async def test_missing_api_key(
        self: "TestApiKeyValidation",
        mock_settings: MagicMock,
    ) -> None:
        mock_settings.openai_api_key = None

        with pytest.raises(LLMError, match="OPENAI_API_KEY is not configured"):
            OpenAIClient(settings=mock_settings)
