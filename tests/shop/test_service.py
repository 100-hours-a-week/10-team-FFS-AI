from unittest.mock import ANY, AsyncMock, MagicMock

import pytest

from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopOutfit,
    ShopParsedQuery,
    ShopProduct,
    ShopSearchQuery,
    ShopSearchRequest,
    ShopSearchResponse,
)
from app.shop.service import ShopService


@pytest.fixture
def mock_query_parser() -> MagicMock:
    parser = MagicMock()
    parser.parse = AsyncMock(
        return_value=ShopParsedQuery(
            occasion="일상",
            style="Y2K",
            price_max=30000,
            constraints=["크롭탑"],
        )
    )
    return parser


@pytest.fixture
def mock_search_builder() -> MagicMock:
    builder = MagicMock()
    builder.build = MagicMock(
        return_value=[
            ShopSearchQuery(
                text="TOP Y2K", category_filter="TOP"
            ),
            ShopSearchQuery(
                text="BOTTOM Y2K",
                category_filter="BOTTOM",
            ),
        ]
    )
    return builder


@pytest.fixture
def mock_repository() -> MagicMock:
    repo = MagicMock()
    repo.search_multiple = AsyncMock(
        return_value=[
            ProductSearchResult(
                category="TOP",
                candidates=[
                    ProductCandidate(
                        product_id="prod_001",
                        title="Y2K 크롭탑",
                        brand="무신사",
                        price=29000,
                        image_url="https://img.com/1.jpg",
                        link="https://musinsa.com/1",
                        source="musinsa",
                        category="TOP",
                        similarity_score=0.95,
                    )
                ],
            ),
            ProductSearchResult(
                category="BOTTOM",
                candidates=[
                    ProductCandidate(
                        product_id="prod_002",
                        title="와이드 데님",
                        brand="무신사",
                        price=45000,
                        image_url="https://img.com/2.jpg",
                        link="https://musinsa.com/2",
                        source="musinsa",
                        category="BOTTOM",
                        similarity_score=0.90,
                    )
                ],
            ),
        ]
    )
    return repo


@pytest.fixture
def mock_composer() -> MagicMock:
    composer = MagicMock()
    composer.compose = AsyncMock(
        return_value=ShopSearchResponse(
            query_summary="Y2K 크롭탑 코디",
            outfits=[
                ShopOutfit(
                    outfit_id="outfit_s001",
                    items=[
                        ShopProduct(
                            product_id="prod_001",
                            title="Y2K 크롭탑",
                            brand="무신사",
                            price=29000,
                            image_url="https://img.com/1.jpg",
                            link="https://musinsa.com/1",
                            source="musinsa",
                            category="TOP",
                        ),
                        ShopProduct(
                            product_id="prod_002",
                            title="와이드 데님",
                            brand="무신사",
                            price=45000,
                            image_url="https://img.com/2.jpg",
                            link="https://musinsa.com/2",
                            source="musinsa",
                            category="BOTTOM",
                        ),
                    ],
                )
            ],
        )
    )
    return composer


@pytest.fixture
def service(
    mock_query_parser: MagicMock,
    mock_search_builder: MagicMock,
    mock_repository: MagicMock,
    mock_composer: MagicMock,
) -> ShopService:
    return ShopService(
        query_parser=mock_query_parser,
        search_builder=mock_search_builder,
        repository=mock_repository,
        composer=mock_composer,
    )


class TestShopServiceSearch:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        service: ShopService,
        mock_query_parser: MagicMock,
        mock_search_builder: MagicMock,
        mock_repository: MagicMock,
        mock_composer: MagicMock,
    ) -> None:
        """전체 파이프라인 정상 동작"""
        request = ShopSearchRequest(
            user_id=123,
            query="3만원 이하 Y2K 크롭탑 코디",
        )

        response = await service.search(request)

        # 각 컴포넌트 호출 확인
        mock_query_parser.parse.assert_awaited_once_with(
            "3만원 이하 Y2K 크롭탑 코디",
            trace_id=ANY,
            user_id=123,
        )
        mock_search_builder.build.assert_called_once()
        mock_repository.search_multiple.assert_awaited_once()
        mock_composer.compose.assert_awaited_once()

        assert response.query_summary == "Y2K 크롭탑 코디"
        assert len(response.outfits) == 1

    @pytest.mark.asyncio
    async def test_session_id_propagated(
        self,
        service: ShopService,
    ) -> None:
        """session_id가 응답에 전달됨"""
        request = ShopSearchRequest(
            user_id=123,
            query="코디 추천",
            session_id="sess_001",
        )

        response = await service.search(request)

        assert response.session_id == "sess_001"

    @pytest.mark.asyncio
    async def test_repository_receives_parsed_query(
        self,
        service: ShopService,
        mock_repository: MagicMock,
    ) -> None:
        """repository에 parsed 쿼리가 전달됨"""
        request = ShopSearchRequest(
            user_id=123,
            query="코디 추천",
        )

        await service.search(request)

        call_args = (
            mock_repository.search_multiple.call_args
        )
        assert "parsed" in call_args.kwargs
        assert "queries" in call_args.kwargs

    @pytest.mark.asyncio
    async def test_no_user_id_in_repository_call(
        self,
        service: ShopService,
        mock_repository: MagicMock,
    ) -> None:
        """repository 호출에 user_id가 포함되지 않음"""
        request = ShopSearchRequest(
            user_id=123,
            query="코디 추천",
        )

        await service.search(request)

        call_args = (
            mock_repository.search_multiple.call_args
        )
        # repository.search_multiple에 user_id가 인자로 전달되지 않음
        assert "user_id" not in call_args.kwargs
