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
                "category": "TOP",
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
        assert candidate.category == "TOP"
        assert candidate.color == ["검정"]
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

    @pytest.mark.asyncio
    async def test_search_multiple_removes_duplicates_by_clothes_id(
        self,
        repository: ClothingRepository,
        mock_qdrant_client: MagicMock,
    ) -> None:
        """같은 카테고리 내에서 clothes_id가 중복되면 병합하여 1개만 반환한다."""
        # Query 1 결과 : clothes_id=1, 2
        hit1 = ScoredPoint(
            id=1, version=1, score=0.9, payload={"clothesId": 1, "category": "TOP"}
        )
        hit2 = ScoredPoint(
            id=2, version=1, score=0.8, payload={"clothesId": 2, "category": "TOP"}
        )
        # Query 2 결과 : clothes_id=2 (중복), 3
        hit3 = ScoredPoint(
            id=3, version=1, score=0.85, payload={"clothesId": 2, "category": "TOP"}
        )
        hit4 = ScoredPoint(
            id=4, version=1, score=0.7, payload={"clothesId": 3, "category": "TOP"}
        )

        mock_qdrant_client.query_points.side_effect = [
            QueryResponse(points=[hit1, hit2]),
            QueryResponse(points=[hit3, hit4]),
        ]

        queries = [
            SearchQuery(text="q1", category_filter="TOP"),
            SearchQuery(text="q2", category_filter="TOP"),
        ]

        results = await repository.search_multiple(user_id="user123", queries=queries)

        assert len(results) == 1
        assert results[0].category == "TOP"
        # 1, 2, 3 총 3개의 유니크한 clothes_id만 남아야 함
        clothes_ids = {c.clothes_id for c in results[0].candidates}
        assert clothes_ids == {1, 2, 3}
        assert len(results[0].candidates) == 3


class TestToCandidateStatic:
    def test_to_candidate_with_full_payload(self) -> None:
        hit = ScoredPoint(
            id=1,
            version=1,
            score=0.88,
            payload={
                "clothesId": 456,
                "imageUrl": "https://img.com/456.jpg",
                "category": "BOTTOM",
                "subCategory": "슬랙스_트라우저",
                "color": "네이비",
                "styleTags": ["포멀"],
                "caption": "네이비 슬랙스",
            },
        )

        candidate = ClothingRepository._to_candidate(hit)

        assert candidate.clothes_id == 456
        assert candidate.image_url == "https://img.com/456.jpg"
        assert candidate.category == "BOTTOM"
        assert candidate.sub_category == "슬랙스_트라우저"
        assert candidate.color == ["네이비"]
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
                "category": "ETC",
            },
        )

        candidate = ClothingRepository._to_candidate(hit)

        assert candidate.clothes_id == 789
        assert candidate.color == []
        assert candidate.style_tags == []
        assert candidate.caption is None
