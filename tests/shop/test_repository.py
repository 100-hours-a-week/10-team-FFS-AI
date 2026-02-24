from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http.models import ScoredPoint

from app.shop.repository import ShopProductRepository
from app.shop.schemas import (
    ShopParsedQuery,
    ShopSearchQuery,
)


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock()
    service.get_embedding = AsyncMock(
        return_value=[0.1] * 4096
    )
    return service


@pytest.fixture
def mock_qdrant() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def repo(
    mock_embedding_service: MagicMock,
    mock_qdrant: AsyncMock,
) -> ShopProductRepository:
    return ShopProductRepository(
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant,
    )


def _make_scored_point(
    point_id: int,
    payload: dict,
    score: float = 0.9,
) -> ScoredPoint:
    """테스트용 ScoredPoint 생성"""
    return ScoredPoint(
        id=point_id,
        version=0,
        score=score,
        payload=payload,
    )


class TestShopProductRepository:
    @pytest.mark.asyncio
    async def test_search_without_user_id_filter(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """user_id 필터 없이 검색 확인"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[
                _make_scored_point(
                    1,
                    {
                        "productId": "prod_001",
                        "title": "크롭탑",
                        "price": 29000,
                        "category": "TOP",
                    },
                ),
            ]
        )

        query = ShopSearchQuery(
            text="Y2K 상의", category_filter="TOP"
        )
        parsed = ShopParsedQuery(style="Y2K")

        result = await repo.search_by_query(query, parsed)

        # user_id 필터가 포함되지 않았는지 확인
        call_args = mock_qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")
        if query_filter and query_filter.must:
            filter_keys = [
                cond.key for cond in query_filter.must
            ]
            assert "userId" not in filter_keys

        assert len(result.candidates) == 1
        assert result.candidates[0].product_id == "prod_001"

    @pytest.mark.asyncio
    async def test_search_with_price_filter(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """가격 필터 적용 확인"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[]
        )

        query = ShopSearchQuery(
            text="상의", category_filter="TOP"
        )
        parsed = ShopParsedQuery(
            style="캐주얼",
            price_max=30000,
            price_min=10000,
        )

        await repo.search_by_query(query, parsed)

        call_args = mock_qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")

        # 가격 필터가 포함되었는지 확인
        assert query_filter is not None
        price_conditions = [
            c
            for c in query_filter.must
            if c.key == "price"
        ]
        assert len(price_conditions) == 1
        assert price_conditions[0].range.gte == 10000
        assert price_conditions[0].range.lte == 30000

    @pytest.mark.asyncio
    async def test_search_with_brand_filter(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """브랜드 필터 적용 확인"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[]
        )

        query = ShopSearchQuery(
            text="상의", category_filter="TOP"
        )
        parsed = ShopParsedQuery(
            style="캐주얼", brand="나이키"
        )

        await repo.search_by_query(query, parsed)

        call_args = mock_qdrant.query_points.call_args
        query_filter = call_args.kwargs.get("query_filter")

        brand_conditions = [
            c
            for c in query_filter.must
            if c.key == "brand"
        ]
        assert len(brand_conditions) == 1

    @pytest.mark.asyncio
    async def test_search_no_filters(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """필터 조건 없는 검색"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[]
        )

        query = ShopSearchQuery(text="아무 옷")
        parsed = ShopParsedQuery()

        await repo.search_by_query(query, parsed)

        call_args = mock_qdrant.query_points.call_args
        # 필터 조건이 없으면 query_filter는 None
        assert call_args.kwargs.get("query_filter") is None

    @pytest.mark.asyncio
    async def test_search_empty_results(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """빈 결과 처리"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[]
        )

        query = ShopSearchQuery(
            text="희귀한 옷", category_filter="TOP"
        )
        parsed = ShopParsedQuery()

        result = await repo.search_by_query(query, parsed)

        assert result.candidates == []
        assert result.category == "TOP"

    @pytest.mark.asyncio
    async def test_to_candidate_mapping(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """ScoredPoint → ProductCandidate 매핑 확인"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[
                _make_scored_point(
                    1,
                    {
                        "productId": "prod_001",
                        "title": "Y2K 크롭탑",
                        "brand": "무신사",
                        "price": 29000,
                        "imageUrl": "https://img.com/1.jpg",
                        "link": "https://musinsa.com/1",
                        "source": "musinsa",
                        "category": "TOP",
                        "styleTags": ["Y2K", "크롭"],
                    },
                    score=0.95,
                ),
            ]
        )

        query = ShopSearchQuery(
            text="상의", category_filter="TOP"
        )
        parsed = ShopParsedQuery()

        result = await repo.search_by_query(query, parsed)

        c = result.candidates[0]
        assert c.product_id == "prod_001"
        assert c.title == "Y2K 크롭탑"
        assert c.brand == "무신사"
        assert c.price == 29000
        assert c.image_url == "https://img.com/1.jpg"
        assert c.link == "https://musinsa.com/1"
        assert c.source == "musinsa"
        assert c.style_tags == ["Y2K", "크롭"]
        assert c.similarity_score == 0.95

    @pytest.mark.asyncio
    async def test_search_multiple(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """여러 카테고리 동시 검색"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[
                _make_scored_point(
                    1,
                    {
                        "productId": "p1",
                        "title": "상의",
                        "category": "TOP",
                    },
                ),
            ]
        )

        queries = [
            ShopSearchQuery(
                text="TOP", category_filter="TOP"
            ),
            ShopSearchQuery(
                text="BOTTOM", category_filter="BOTTOM"
            ),
        ]
        parsed = ShopParsedQuery()

        results = await repo.search_multiple(
            queries=queries, parsed=parsed
        )

        assert len(results) == 2
        # 2번 호출 확인 (asyncio.gather)
        assert mock_qdrant.query_points.call_count == 2

    @pytest.mark.asyncio
    async def test_uses_shop_collection(
        self,
        repo: ShopProductRepository,
        mock_qdrant: AsyncMock,
    ) -> None:
        """shop_products 컬렉션 사용 확인"""
        mock_qdrant.query_points.return_value = MagicMock(
            points=[]
        )

        query = ShopSearchQuery(text="상의")
        parsed = ShopParsedQuery()

        await repo.search_by_query(query, parsed)

        call_args = mock_qdrant.query_points.call_args
        assert (
            call_args.kwargs["collection_name"]
            == repo.settings.qdrant_shop_collection_name
        )
