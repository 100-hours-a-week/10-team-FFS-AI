from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.http.models import QueryResponse, ScoredPoint

from app.outfit.repository import ClothingRepository
from app.outfit.schemas import SearchQuery


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    service = MagicMock()
    service.get_embedding = AsyncMock(return_value=[0.1] * 4096)
    return service


@pytest.fixture
def mock_qdrant_client() -> MagicMock:
    client = MagicMock()
    # Qdrant 1.7+ API: query_points 반환 형태
    client.query_points = AsyncMock(return_value=QueryResponse(points=[]))
    return client


@pytest.fixture
def repository(
    mock_embedding_service: MagicMock,
    mock_qdrant_client: MagicMock,
) -> ClothingRepository:
    return ClothingRepository(
        embedding_service=mock_embedding_service,
        qdrant_client=mock_qdrant_client,
    )


class TestSearchByQuery:
    @pytest.mark.asyncio
    async def test_search_with_category_filter(
        self,
        repository: ClothingRepository,
        mock_qdrant_client: MagicMock,
    ) -> None:
        query = SearchQuery(text="캐주얼 상의", category_filter="상의")

        result = await repository.search_by_query(
            user_id="user123",
            query=query,
            top_k=5,
        )

        assert result.category == "상의"
        assert result.candidates == []
        mock_qdrant_client.query_points.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_without_category_filter(
        self,
        repository: ClothingRepository,
        mock_qdrant_client: MagicMock,
    ) -> None:
        query = SearchQuery(text="면접용 옷")

        result = await repository.search_by_query(
            user_id="user123",
            query=query,
        )

        assert result.category == "전체"
        mock_qdrant_client.query_points.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_search_returns_candidates(
        self,
        repository: ClothingRepository,
        mock_qdrant_client: MagicMock,
    ) -> None:
        mock_hit = ScoredPoint(
            id=1,
            version=1,
            score=0.95,
            payload={
                "clothesId": 123,
                "imageUrl": "https://example.com/img.jpg",
                "category": "상의",
                "color": "검정",
                "styleTags": ["캐주얼", "베이직"],
                "caption": "검은색 반팔 티셔츠",
            },
        )
        mock_qdrant_client.query_points = AsyncMock(
            return_value=QueryResponse(points=[mock_hit])
        )

        query = SearchQuery(text="검정 티셔츠", category_filter="상의")

        result = await repository.search_by_query(
            user_id="user123",
            query=query,
        )

        assert len(result.candidates) == 1
        candidate = result.candidates[0]
        assert candidate.clothes_id == 123
        assert candidate.category == "상의"
        assert candidate.color == "검정"
        assert candidate.similarity_score == 0.95


class TestSearchMultiple:
    @pytest.mark.asyncio
    async def test_search_multiple_categories(
        self,
        repository: ClothingRepository,
        mock_qdrant_client: MagicMock,
    ) -> None:
        queries = [
            SearchQuery(text="상의 검색", category_filter="상의"),
            SearchQuery(text="하의 검색", category_filter="하의"),
            SearchQuery(text="아우터 검색", category_filter="아우터"),
        ]

        results = await repository.search_multiple(
            user_id="user123",
            queries=queries,
        )

        assert len(results) == 3
        assert mock_qdrant_client.query_points.await_count == 3


class TestToCandidateStatic:
    def test_to_candidate_with_full_payload(self) -> None:
        hit = ScoredPoint(
            id=1,
            version=1,
            score=0.88,
            payload={
                "clothesId": 456,
                "imageUrl": "https://img.com/456.jpg",
                "category": "하의",
                "color": "네이비",
                "styleTags": ["포멀"],
                "caption": "네이비 슬랙스",
            },
        )

        candidate = ClothingRepository._to_candidate(hit)

        assert candidate.clothes_id == 456
        assert candidate.image_url == "https://img.com/456.jpg"
        assert candidate.category == "하의"
        assert candidate.color == "네이비"
        assert candidate.style_tags == ["포멀"]
        assert candidate.caption == "네이비 슬랙스"
        assert candidate.similarity_score == 0.88

    def test_to_candidate_with_missing_optional_fields(self) -> None:
        hit = ScoredPoint(
            id=2,
            version=1,
            score=0.75,
            payload={
                "clothesId": 789,
                "imageUrl": "https://img.com/789.jpg",
                "category": "아우터",
            },
        )

        candidate = ClothingRepository._to_candidate(hit)

        assert candidate.clothes_id == 789
        assert candidate.color is None
        assert candidate.style_tags == []
        assert candidate.caption is None
