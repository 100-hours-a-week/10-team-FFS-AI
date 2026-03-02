from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.llm_schemas import ShopQueryLLMResponse
from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.query_parser import ShopQueryParser
from app.shop.schemas import ShopParsedQuery


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def parser(mock_llm: MagicMock) -> ShopQueryParser:
    return ShopQueryParser(llm_client=mock_llm)


class TestShopQueryParser:
    @pytest.mark.asyncio
    async def test_parse_basic_query(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopQueryLLMResponse(
                occasion="일상",
                style="Y2K",
                price_max=30000,
                constraints=["크롭탑"],
            )
        )

        result = await parser.parse("3만원 이하 Y2K 크롭탑 코디")

        assert isinstance(result, ShopParsedQuery)
        assert result.style == "Y2K"
        assert result.price_max == 30000
        assert result.constraints == ["크롭탑"]

    @pytest.mark.asyncio
    async def test_parse_with_brand(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopQueryLLMResponse(
                occasion="일상",
                style="캐주얼",
                brand="나이키",
            )
        )

        result = await parser.parse("나이키 캐주얼 코디")

        assert result.brand == "나이키"
        assert result.style == "캐주얼"

    @pytest.mark.asyncio
    async def test_parse_defaults(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(return_value=ShopQueryLLMResponse())

        result = await parser.parse("아무 옷")

        assert result.occasion == "일상"
        assert result.style == "깔끔한"
        assert result.price_max is None
        assert result.constraints == []

    @pytest.mark.asyncio
    async def test_parse_with_price_range(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopQueryLLMResponse(
                occasion="일상",
                style="캐주얼",
                price_min=10000,
                price_max=50000,
            )
        )

        result = await parser.parse("1만원~5만원 캐주얼 코디")

        assert result.price_min == 10000
        assert result.price_max == 50000

    @pytest.mark.asyncio
    async def test_parse_with_target_category(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopQueryLLMResponse(
                style="캐주얼",
                target_category="TOP",
            )
        )

        result = await parser.parse("캐주얼 상의 추천")

        assert result.target_category == "TOP"
        assert result.is_full_outfit_request() is False

    @pytest.mark.asyncio
    async def test_response_format_passed_to_llm(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(return_value=ShopQueryLLMResponse())

        await parser.parse("코디 추천")

        call_kwargs = mock_llm.chat_completion.call_args.kwargs
        assert call_kwargs["response_format"] is ShopQueryLLMResponse
        assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_llm_error_propagates(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(side_effect=ShopLLMError("API 장애"))

        with pytest.raises(ShopLLMError):
            await parser.parse("테스트")

    @pytest.mark.asyncio
    async def test_unexpected_error_raises_parse_error(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            side_effect=RuntimeError("예상치 못한 에러")
        )

        with pytest.raises(ShopParseError, match="Unexpected parsing error"):
            await parser.parse("테스트")
