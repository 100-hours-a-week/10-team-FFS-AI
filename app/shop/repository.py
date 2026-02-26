import asyncio
import logging

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import ScoredPoint

from app.common.metrics import measure_time
from app.config import Settings, get_settings
from app.core.database import get_qdrant_client
from app.embedding.service import EmbeddingService, get_embedding_service
from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopParsedQuery,
    ShopSearchQuery,
)

logger = logging.getLogger(__name__)


class ShopProductRepository:
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
        query: ShopSearchQuery,
        parsed: ShopParsedQuery,
        top_k: int = 5,
    ) -> ProductSearchResult:
        embedding_service = await self._get_embedding_service()
        qdrant = await self._get_qdrant()

        query_vector = await embedding_service.get_embedding(query.text)

        must_conditions: list[qdrant_models.Condition] = []

        if query.category_filter:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="category",
                    match=qdrant_models.MatchValue(value=query.category_filter),
                )
            )

        price_range = self._build_price_range(parsed)
        if price_range:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="price",
                    range=price_range,
                )
            )

        if parsed.brand:
            must_conditions.append(
                qdrant_models.FieldCondition(
                    key="brand",
                    match=qdrant_models.MatchValue(value=parsed.brand),
                )
            )

        query_filter = (
            qdrant_models.Filter(must=must_conditions) if must_conditions else None
        )

        response = await qdrant.query_points(
            collection_name=self.settings.qdrant_shop_collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        candidates = [self._to_candidate(hit) for hit in response.points]
        category = query.category_filter or "전체"

        logger.info(
            "Shop search completed: category=%s, found=%d",
            category,
            len(candidates),
        )

        return ProductSearchResult(category=category, candidates=candidates)

    @measure_time("shop_vector_search")
    async def search_multiple(
        self,
        queries: list[ShopSearchQuery],
        parsed: ShopParsedQuery,
        top_k: int = 5,
        trace_id: str | None = None,
    ) -> list[ProductSearchResult]:
        log_context = f"trace_id={trace_id} " if trace_id else ""
        logger.info(
            f"Searching shop products | {log_context}" f"query_count={len(queries)}"
        )

        tasks = [self.search_by_query(query, parsed, top_k) for query in queries]
        results = await asyncio.gather(*tasks)

        total_found = sum(len(r.candidates) for r in results)
        logger.info(
            f"Shop search completed | {log_context}" f"total_candidates={total_found}"
        )

        return results

    @staticmethod
    def _build_price_range(
        parsed: ShopParsedQuery,
    ) -> qdrant_models.Range | None:
        if parsed.price_min is None and parsed.price_max is None:
            return None

        return qdrant_models.Range(
            gte=parsed.price_min,
            lte=parsed.price_max,
        )

    @staticmethod
    def _to_candidate(hit: ScoredPoint) -> ProductCandidate:
        payload = hit.payload or {}

        def get_str(key: str, default: str = "") -> str:
            val = payload.get(key)
            return str(val) if val is not None else default

        return ProductCandidate(
            product_id=get_str("productId", str(hit.id)),
            title=get_str("title"),
            brand=get_str("brand"),
            price=payload.get("price") or 0,
            image_url=get_str("imageUrl"),
            link=get_str("link"),
            source=get_str("source"),
            category=get_str("category"),
            style_tags=payload.get("styleTags") or [],
            similarity_score=hit.score,
        )
