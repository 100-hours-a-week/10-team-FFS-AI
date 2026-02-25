import asyncio
import logging

from langfuse.decorators import observe
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import ScoredPoint

from app.common.metrics import measure_time
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
        user_id: int,
        query: SearchQuery,
        top_k: int = 5,
    ) -> SearchResult:
        embedding_service = await self._get_embedding_service()
        qdrant = await self._get_qdrant()

        query_vector = await embedding_service.get_embedding(query.text)

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

        response = await qdrant.query_points(
            collection_name=self.settings.qdrant_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        candidates = [self._to_candidate(hit) for hit in response.points]
        category = query.category_filter or "전체"

        logger.info(
            "Search completed: user_id=%s, category=%s, found=%d",
            user_id,
            category,
            len(candidates),
        )

        return SearchResult(category=category, candidates=candidates)

    @observe(name="outfit_vector_search")
    @measure_time("vector_search")
    async def search_multiple(
        self,
        user_id: int,
        queries: list[SearchQuery],
        top_k: int = 5,
        trace_id: str | None = None,
    ) -> list[SearchResult]:
        log_context = f"trace_id={trace_id} " if trace_id else ""
        logger.info(
            f"Searching multiple queries | {log_context}"
            f"user_id={user_id} query_count={len(queries)}"
        )

        tasks = [self.search_by_query(user_id, query, top_k) for query in queries]
        results = await asyncio.gather(*tasks)

        total_found = sum(len(r.candidates) for r in results)
        logger.info(
            f"Search completed | {log_context}"
            f"user_id={user_id} total_candidates={total_found}"
        )

        return results

    @staticmethod
    def _to_candidate(hit: ScoredPoint) -> ClothingCandidate:
        payload = hit.payload or {}

        raw_color = payload.get("color")
        if raw_color is None:
            color_list: list[str] = []

        elif isinstance(raw_color, str):
            color_list = [raw_color] if raw_color else []
        else:
            color_list = list(raw_color)
        return ClothingCandidate(
            clothes_id=payload.get("clothesId", 0),
            image_url=payload.get("imageUrl", ""),
            category=payload.get("category", ""),
            color=color_list,
            style_tags=payload.get("styleTags", []),
            caption=payload.get("caption"),
            similarity_score=hit.score,
        )
