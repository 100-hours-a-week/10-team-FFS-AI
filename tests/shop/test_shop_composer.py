from unittest.mock import AsyncMock, MagicMock

import pytest

from app.common.llm_schemas import (
    ShopCombination,
    ShopCompositionLLMResponse,
    ShopItemLLM,
)
from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopParsedQuery,
)
from app.shop.shop_composer import ShopComposer


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
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopCompositionLLMResponse(
                query_summary="Y2K 크롭탑 코디",
                outfits=[
                    ShopCombination(
                        items=[
                            ShopItemLLM(product_id="prod_001"),
                            ShopItemLLM(product_id="prod_002"),
                        ]
                    )
                ],
            )
        )

        parsed = ShopParsedQuery(style="Y2K")
        response = await composer.compose(
            parsed_query=parsed, search_results=_make_candidates()
        )

        assert response.query_summary == "Y2K 크롭탑 코디"
        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 2
        item = response.outfits[0].items[0]
        assert item.title == "Y2K 크롭탑"
        assert item.price == 29000
        assert item.brand == "무신사스탠다드"

    @pytest.mark.asyncio
    async def test_compose_empty_candidates(self, composer: ShopComposer) -> None:
        parsed = ShopParsedQuery(style="Y2K")
        response = await composer.compose(parsed_query=parsed, search_results=[])

        assert response.query_summary == "검색 결과가 없습니다"
        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_all_empty_candidates(self, composer: ShopComposer) -> None:
        parsed = ShopParsedQuery(style="Y2K")
        results = [
            ProductSearchResult(category="TOP", candidates=[]),
            ProductSearchResult(category="BOTTOM", candidates=[]),
        ]
        response = await composer.compose(parsed_query=parsed, search_results=results)
        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_ignores_invalid_product_id(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopCompositionLLMResponse(
                query_summary="코디 추천",
                outfits=[
                    ShopCombination(
                        items=[
                            ShopItemLLM(product_id="prod_001"),
                            ShopItemLLM(product_id="INVALID_ID"),
                        ]
                    )
                ],
            )
        )

        response = await composer.compose(
            parsed_query=ShopParsedQuery(style="캐주얼"),
            search_results=_make_candidates(),
        )

        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 1
        assert response.outfits[0].items[0].product_id == "prod_001"

    @pytest.mark.asyncio
    async def test_compose_all_invalid_ids_empty_outfit(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopCompositionLLMResponse(
                query_summary="코디 추천",
                outfits=[
                    ShopCombination(
                        items=[
                            ShopItemLLM(product_id="INVALID_1"),
                            ShopItemLLM(product_id="INVALID_2"),
                        ]
                    )
                ],
            )
        )

        response = await composer.compose(
            parsed_query=ShopParsedQuery(style="캐주얼"),
            search_results=_make_candidates(),
        )
        assert response.outfits == []

    @pytest.mark.asyncio
    async def test_compose_llm_error_propagates(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(side_effect=ShopLLMError("API 장애"))

        with pytest.raises(ShopLLMError):
            await composer.compose(
                parsed_query=ShopParsedQuery(style="캐주얼"),
                search_results=_make_candidates(),
            )

    @pytest.mark.asyncio
    async def test_compose_unexpected_error_raises_parse_error(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            side_effect=RuntimeError("예상치 못한 에러")
        )

        with pytest.raises(ShopParseError):
            await composer.compose(
                parsed_query=ShopParsedQuery(style="캐주얼"),
                search_results=_make_candidates(),
            )

    @pytest.mark.asyncio
    async def test_compose_prompt_includes_price(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopCompositionLLMResponse(query_summary="코디", outfits=[])
        )

        await composer.compose(
            parsed_query=ShopParsedQuery(style="캐주얼", price_max=30000),
            search_results=_make_candidates(),
        )

        call_args = mock_llm.chat_completion.call_args
        user_msg = call_args.kwargs["messages"][1]["content"]
        assert "30,000" in user_msg

    @pytest.mark.asyncio
    async def test_compose_multiple_outfits(
        self, composer: ShopComposer, mock_llm: MagicMock
    ) -> None:
        mock_llm.chat_completion = AsyncMock(
            return_value=ShopCompositionLLMResponse(
                query_summary="코디 추천",
                outfits=[
                    ShopCombination(items=[ShopItemLLM(product_id="prod_001")]),
                    ShopCombination(items=[ShopItemLLM(product_id="prod_002")]),
                ],
            )
        )

        response = await composer.compose(
            parsed_query=ShopParsedQuery(style="캐주얼"),
            search_results=_make_candidates(),
        )
        assert len(response.outfits) == 2
