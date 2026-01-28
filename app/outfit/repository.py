import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import ScoredPoint

from app.config import Settings, get_settings
from app.core.database import get_qdrant_client
from app.embedding.service import EmbeddingService, get_embedding_service
from app.outfit.schemas import ClothingCandidate, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class ClothingRepository:
    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        qdrant_client: AsyncQdrantClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._embedding_service = embedding_service
        self._qdrant_client = qdrant_client
        self.settings = settings or get_settings()

    async def _get_embedding_service(self) -> EmbeddingService:
        if self._embedding_service is None:
            self._embedding_service = get_embedding_service()
        return self._embedding_service

    async def _get_qdrant(self) -> AsyncQdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = await get_qdrant_client()
        return self._qdrant_client

    async def search_by_query(
        self,
        user_id: str,
        query: SearchQuery,
        top_k: int = 5,
    ) -> SearchResult:
        embedding_service = await self._get_embedding_service()
        qdrant = await self._get_qdrant()

        query_vector = await embedding_service.get_embedding(query.text)

        # 필터 구성: user_id 필수 + category 선택
        must_conditions: list[qdrant_models.Condition] = [
            qdrant_models.FieldCondition(
                key="userId",
                match=qdrant_models.MatchValue(value=user_id),
            )
        ]

        if query.category_filter:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="category",
                    match=qdrant_models.MatchValue(value=query.category_filter),
                )
            )

        query_filter = qdrant_models.Filter(must=must_conditions)

        results = await qdrant.search(
            collection_name=self.settings.qdrant_collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        candidates = [self._to_candidate(hit) for hit in results]
        category = query.category_filter or "전체"

        logger.info(
            "Search completed: user_id=%s, category=%s, found=%d",
            user_id,
            category,
            len(candidates),
        )

        return SearchResult(category=category, candidates=candidates)

    async def search_multiple(
        self,
        user_id: str,
        queries: list[SearchQuery],
        top_k: int = 5,
    ) -> list[SearchResult]:
        tasks = [self.search_by_query(user_id, query, top_k) for query in queries]
        return await asyncio.gather(*tasks)

    @staticmethod
    def _to_candidate(hit: ScoredPoint) -> ClothingCandidate:
        payload = hit.payload or {}
        return ClothingCandidate(
            clothes_id=payload.get("clothesId", 0),
            image_url=payload.get("imageUrl", ""),
            category=payload.get("category", ""),
            color=payload.get("color"),
            style_tags=payload.get("styleTags", []),
            caption=payload.get("caption"),
            similarity_score=hit.score,
        )
