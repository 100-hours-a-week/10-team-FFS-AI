import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.query_parser import ShopQueryParser
from app.shop.schemas import ShopParsedQuery


def _make_llm_response(data: dict[str, Any]) -> dict[str, Any]:
    """LLM 응답 형식으로 래핑"""
    return {
        "choices": [
            {"message": {"content": json.dumps(data, ensure_ascii=False)}}
        ]
    }


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
        """기본 쿼리 파싱 확인"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "occasion": "일상",
                    "style": "Y2K",
                    "season": None,
                    "price_max": 30000,
                    "price_min": None,
                    "brand": None,
                    "target_category": None,
                    "constraints": ["크롭탑"],
                }
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
        """브랜드 포함 쿼리 파싱"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "occasion": "일상",
                    "style": "캐주얼",
                    "brand": "나이키",
                    "price_max": None,
                    "price_min": None,
                    "constraints": [],
                }
            )
        )

        result = await parser.parse("나이키 캐주얼 코디")

        assert result.brand == "나이키"
        assert result.style == "캐주얼"

    @pytest.mark.asyncio
    async def test_parse_defaults(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """필드 누락 시 기본값 확인"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response({})
        )

        result = await parser.parse("아무 옷")

        assert result.occasion == "일상"
        assert result.style == "깔끔한"
        assert result.price_max is None
        assert result.constraints == []

    @pytest.mark.asyncio
    async def test_parse_markdown_wrapped_json(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """마크다운 코드블록으로 감싼 JSON 처리"""
        wrapped = '```json\n{"occasion": "데이트", "style": "로맨틱"}\n```'
        mock_llm.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": wrapped}}]
            }
        )

        result = await parser.parse("데이트 코디")

        assert result.occasion == "데이트"
        assert result.style == "로맨틱"

    @pytest.mark.asyncio
    async def test_parse_invalid_json_raises_error(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """잘못된 JSON 응답 시 ShopParseError 발생"""
        mock_llm.chat_completion = AsyncMock(
            return_value={
                "choices": [{"message": {"content": "이건 JSON이 아닙니다"}}]
            }
        )

        with pytest.raises(ShopParseError):
            await parser.parse("테스트")

    @pytest.mark.asyncio
    async def test_parse_llm_error_propagation(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """LLMError는 ShopLLMError가 아니므로 ShopParseError로 래핑"""
        mock_llm.chat_completion = AsyncMock(
            side_effect=Exception("API 장애")
        )

        with pytest.raises(ShopParseError):
            await parser.parse("테스트")

    @pytest.mark.asyncio
    async def test_parse_with_price_range(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """가격 범위 파싱"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "occasion": "일상",
                    "style": "캐주얼",
                    "price_min": 10000,
                    "price_max": 50000,
                }
            )
        )

        result = await parser.parse("1만원~5만원 캐주얼 코디")

        assert result.price_min == 10000
        assert result.price_max == 50000

    @pytest.mark.asyncio
    async def test_parse_with_target_category(
        self, parser: ShopQueryParser, mock_llm: MagicMock
    ) -> None:
        """특정 카테고리 요청 파싱"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "style": "캐주얼",
                    "target_category": "TOP",
                }
            )
        )

        result = await parser.parse("캐주얼 상의 추천")

        assert result.target_category == "TOP"
        assert result.is_full_outfit_request() is False
