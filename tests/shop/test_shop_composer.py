import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.shop.exceptions import ShopParseError
from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopParsedQuery,
)
from app.shop.shop_composer import ShopComposer


def _make_llm_response(data: dict[str, Any]) -> dict[str, Any]:
    return {"choices": [{"message": {"content": json.dumps(data, ensure_ascii=False)}}]}


def _make_candidates() -> list[ProductSearchResult]:
    return [
        ProductSearchResult(
            category="TOP",
            candidates=[
                ProductCandidate(
                    product_id="prod_001",
                    title="Y2K 크롭탑",
                    brand="무신사스탠다드",
                    price=29000,
                    image_url="https://img.com/001.jpg",
                    link="https://musinsa.com/001",
                    source="musinsa",
                    category="TOP",
                    style_tags=["Y2K"],
                    similarity_score=0.95,
                ),
            ],
        ),
        ProductSearchResult(
            category="BOTTOM",
            candidates=[
                ProductCandidate(
                    product_id="prod_002",
                    title="와이드 데님",
                    brand="무신사스탠다드",
                    price=45000,
                    image_url="https://img.com/002.jpg",
                    link="https://musinsa.com/002",
                    source="musinsa",
                    category="BOTTOM",
                    style_tags=["캐주얼"],
                    similarity_score=0.90,
                ),
            ],
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock()


@pytest.fixture
def composer(mock_llm: MagicMock) -> ShopComposer:
    return ShopComposer(llm_client=mock_llm)


class TestShopComposer:
    @pytest.mark.asyncio
    async def test_compose_success(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """정상적인 코디 조합 생성"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "query_summary": "Y2K 크롭탑 코디",
                    "outfits": [
                        {
                            "items": [
                                {"product_id": "prod_001"},
                                {"product_id": "prod_002"},
                            ]
                        }
                    ],
                }
            )
        )

        parsed = ShopParsedQuery(style="Y2K")
        search_results = _make_candidates()

        response = await composer.compose(
            parsed_query=parsed,
            search_results=search_results,
        )

        assert response.query_summary == "Y2K 크롭탑 코디"
        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 2
        # 상품 정보가 채워졌는지 확인
        item = response.outfits[0].items[0]
        assert item.title == "Y2K 크롭탑"
        assert item.price == 29000
        assert item.brand == "무신사스탠다드"

    @pytest.mark.asyncio
    async def test_compose_empty_candidates(self, composer: ShopComposer) -> None:
        """후보 없을 때 빈 응답 반환"""
        parsed = ShopParsedQuery(style="Y2K")
        empty_results: list[ProductSearchResult] = []

        response = await composer.compose(
            parsed_query=parsed,
            search_results=empty_results,
        )

        assert response.query_summary == "검색 결과가 없습니다"
        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_all_empty_candidates(self, composer: ShopComposer) -> None:
        """모든 카테고리 결과가 비어있을 때"""
        parsed = ShopParsedQuery(style="Y2K")
        results = [
            ProductSearchResult(category="TOP", candidates=[]),
            ProductSearchResult(category="BOTTOM", candidates=[]),
        ]

        response = await composer.compose(
            parsed_query=parsed,
            search_results=results,
        )

        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_ignores_invalid_product_id(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """LLM이 존재하지 않는 product_id를 반환하면 무시"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "query_summary": "코디 추천",
                    "outfits": [
                        {
                            "items": [
                                {"product_id": "prod_001"},
                                {"product_id": "INVALID_ID"},
                            ]
                        }
                    ],
                }
            )
        )

        parsed = ShopParsedQuery(style="캐주얼")
        results = _make_candidates()

        response = await composer.compose(
            parsed_query=parsed,
            search_results=results,
        )

        # 유효한 상품만 포함
        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 1
        assert response.outfits[0].items[0].product_id == "prod_001"

    @pytest.mark.asyncio
    async def test_compose_all_invalid_ids_empty_outfit(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """모든 product_id가 잘못되면 해당 outfit 제외"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "query_summary": "코디 추천",
                    "outfits": [
                        {
                            "items": [
                                {"product_id": "INVALID_1"},
                                {"product_id": "INVALID_2"},
                            ]
                        }
                    ],
                }
            )
        )

        parsed = ShopParsedQuery(style="캐주얼")
        results = _make_candidates()

        response = await composer.compose(
            parsed_query=parsed,
            search_results=results,
        )

        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_invalid_json_raises_error(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """잘못된 JSON 응답 시 ShopParseError"""
        mock_llm.chat_completion = AsyncMock(
            return_value={"choices": [{"message": {"content": "이건 JSON이 아닙니다"}}]}
        )

        parsed = ShopParsedQuery(style="캐주얼")
        results = _make_candidates()

        with pytest.raises(ShopParseError):
            await composer.compose(
                parsed_query=parsed,
                search_results=results,
            )

    @pytest.mark.asyncio
    async def test_compose_prompt_includes_price(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """프롬프트에 가격 정보가 포함되는지 확인"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "query_summary": "코디",
                    "outfits": [],
                }
            )
        )

        parsed = ShopParsedQuery(style="캐주얼", price_max=30000)
        results = _make_candidates()

        await composer.compose(
            parsed_query=parsed,
            search_results=results,
        )

        # LLM에 전달된 프롬프트 확인
        call_args = mock_llm.chat_completion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "30,000" in user_msg

    @pytest.mark.asyncio
    async def test_compose_multiple_outfits(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        """여러 코디 조합 생성"""
        mock_llm.chat_completion = AsyncMock(
            return_value=_make_llm_response(
                {
                    "query_summary": "코디 추천",
                    "outfits": [
                        {
                            "items": [
                                {"product_id": "prod_001"},
                            ]
                        },
                        {
                            "items": [
                                {"product_id": "prod_002"},
                            ]
                        },
                    ],
                }
            )
        )

        parsed = ShopParsedQuery(style="캐주얼")
        results = _make_candidates()

        response = await composer.compose(
            parsed_query=parsed,
            search_results=results,
        )

        assert len(response.outfits) == 2
