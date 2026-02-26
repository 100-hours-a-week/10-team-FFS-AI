from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.llm_schemas import OutfitQueryLLMResponse, ReferenceItemLLM
from app.outfit.exceptions import LLMError, ParseError
from app.outfit.query_parser import QueryParser


class TestQueryParserSuccess:
    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm_client: MagicMock) -> QueryParser:
        return QueryParser(mock_llm_client)

    @pytest.mark.asyncio
    async def test_full_outfit_request(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            return_value=OutfitQueryLLMResponse(
                occasion="면접",
                style="포멀",
                season="가을",
                formality="포멀",
                reference_item=None,
                target_category=None,
                constraints=["단정하게"],
            )
        )

        # When
        result = await parser.parse("내일 면접인데 단정하게 입고 싶어")

        # Then
        assert result.occasion == "면접"
        assert result.style == "포멀"
        assert result.season == "가을"
        assert result.formality == "포멀"
        assert result.reference_item is None
        assert result.target_category is None
        assert result.constraints == ["단정하게"]
        assert result.is_full_outfit_request() is True

    @pytest.mark.asyncio
    async def test_matching_request_with_reference_item(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            return_value=OutfitQueryLLMResponse(
                occasion="면접",
                style="포멀",
                season=None,
                formality="포멀",
                reference_item=ReferenceItemLLM(
                    category="TOP",
                    color="검정",
                    style="오버핏",
                    description=None,
                ),
                target_category="BOTTOM",
                constraints=[],
            )
        )

        # When
        result = await parser.parse("검정 코트에 어울리는 바지 추천해줘")

        # Then
        assert result.target_category == "BOTTOM"
        assert result.reference_item is not None
        assert result.reference_item.category == "TOP"
        assert result.reference_item.color == "검정"
        assert result.reference_item.style == "오버핏"
        assert result.is_full_outfit_request() is False
        assert result.is_matching_request() is True

    @pytest.mark.asyncio
    async def test_simple_category_request(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            return_value=OutfitQueryLLMResponse(
                occasion="일상",
                style="깔끔한",
                target_category="BOTTOM",
            )
        )

        # When
        result = await parser.parse("바지 추천해줘")

        # Then
        assert result.target_category == "BOTTOM"
        assert result.reference_item is None
        assert result.is_full_outfit_request() is False
        assert result.is_matching_request() is False

    @pytest.mark.asyncio
    async def test_default_values(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given: 기본값만 있는 응답
        mock_llm_client.chat_completion = AsyncMock(
            return_value=OutfitQueryLLMResponse()
        )

        # When
        result = await parser.parse("뭐 입지")

        # Then
        assert result.occasion == "일상"
        assert result.style == "깔끔한"
        assert result.constraints == []

    @pytest.mark.asyncio
    async def test_response_format_passed_to_llm(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        """chat_completion 호출 시 response_format이 올바르게 전달되는지 검증."""
        mock_llm_client.chat_completion = AsyncMock(
            return_value=OutfitQueryLLMResponse()
        )

        await parser.parse("코디 추천해줘")

        call_kwargs = mock_llm_client.chat_completion.call_args.kwargs
        assert call_kwargs["response_format"] is OutfitQueryLLMResponse
        assert call_kwargs["temperature"] == 0.0


class TestQueryParserFailure:
    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm_client: MagicMock) -> QueryParser:
        return QueryParser(mock_llm_client)

    @pytest.mark.asyncio
    async def test_llm_error_propagates(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            side_effect=LLMError("API key invalid")
        )

        # When / Then
        with pytest.raises(LLMError, match="API key invalid"):
            await parser.parse("코디 추천해줘")

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_parse_error(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            side_effect=RuntimeError("예상치 못한 에러")
        )

        # When / Then
        with pytest.raises(ParseError, match="Unexpected parsing error"):
            await parser.parse("코디 추천해줘")
