import json
from unittest.mock import AsyncMock, MagicMock

import pytest

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
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "occasion": "면접",
                                    "style": "포멀",
                                    "season": "가을",
                                    "formality": "포멀",
                                    "reference_item": None,
                                    "target_category": None,
                                    "constraints": ["단정하게"],
                                }
                            )
                        }
                    }
                ]
            }
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
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "occasion": "면접",
                                    "style": "포멀",
                                    "season": None,
                                    "formality": "포멀",
                                    "reference_item": {
                                        "category": "코트",
                                        "color": "검정",
                                        "style": "오버핏",
                                        "description": None,
                                    },
                                    "target_category": "바지",
                                    "constraints": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

        # When
        result = await parser.parse("검정 코트에 어울리는 바지 추천해줘")

        # Then
        assert result.target_category == "바지"
        assert result.reference_item is not None
        assert result.reference_item.category == "코트"
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
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "occasion": "일상",
                                    "style": "깔끔한",
                                    "season": None,
                                    "formality": None,
                                    "reference_item": None,
                                    "target_category": "바지",
                                    "constraints": [],
                                }
                            )
                        }
                    }
                ]
            }
        )

        # When
        result = await parser.parse("바지 추천해줘")

        # Then
        assert result.target_category == "바지"
        assert result.reference_item is None
        assert result.is_full_outfit_request() is False
        assert result.is_matching_request() is False

    @pytest.mark.asyncio
    async def test_parse_with_markdown_codeblock(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        json_content = json.dumps({"occasion": "데이트", "style": "캐주얼"})
        mock_llm_client.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": f"```json\n{json_content}\n```"}}]
            }
        )

        # When
        result = await parser.parse("데이트 코디 추천해줘")

        # Then
        assert result.occasion == "데이트"
        assert result.style == "캐주얼"

    @pytest.mark.asyncio
    async def test_default_values(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            return_value={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps({})  # 빈 응답
                        }
                    }
                ]
            }
        )

        # When
        result = await parser.parse("뭐 입지")

        # Then
        assert result.occasion == "일상"
        assert result.style == "깔끔한"
        assert result.constraints == []


class TestQueryParserFailure:
    @pytest.fixture
    def mock_llm_client(self) -> MagicMock:
        return MagicMock()

    @pytest.fixture
    def parser(self, mock_llm_client: MagicMock) -> QueryParser:
        return QueryParser(mock_llm_client)

    @pytest.mark.asyncio
    async def test_invalid_json_response(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "면접이니까 정장을 입으세요"}}]
            }
        )

        # When / Then
        with pytest.raises(ParseError, match="Invalid LLM response format"):
            await parser.parse("면접 코디")

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
    async def test_malformed_response_structure(
        self, parser: QueryParser, mock_llm_client: MagicMock
    ) -> None:
        # Given
        mock_llm_client.chat_completion = AsyncMock(return_value={"choices": []})

        # When / Then
        with pytest.raises(ParseError):
            await parser.parse("코디 추천해줘")
