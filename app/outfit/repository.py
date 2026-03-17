import asyncio
import logging

from langfuse import observe
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.models import Record, ScoredPoint

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
        qdrant = await self._get_qdrant()

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

        if not query.text:
            response = await qdrant.scroll(
                collection_name=self.settings.qdrant_collection_name,
                scroll_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )
            points = response[0]
            candidates = [self._to_candidate_from_record(point) for point in points]
        else:
            embedding_service = await self._get_embedding_service()
            query_vector = await embedding_service.get_embedding(query.text)

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

        merged: dict[str, dict[int, ClothingCandidate]] = {}
        for result in results:
            cat = result.category
            if cat not in merged:
                merged[cat] = {}
            for candidate in result.candidates:
                if candidate.clothes_id not in merged[cat]:
                    merged[cat][candidate.clothes_id] = candidate

        final_results = [
            SearchResult(category=cat, candidates=list(cands.values()))
            for cat, cands in merged.items()
        ]

        total_found = sum(len(r.candidates) for r in final_results)
        logger.info(
            f"Search completed | {log_context}"
            f"user_id={user_id} total_candidates={total_found}"
        )

        return final_results

    async def get_by_ids(
        self,
        user_id: int,
        clothes_ids: list[int],
        trace_id: str | None = None,
    ) -> list[ClothingCandidate]:
        """특정 clothes_id 리스트로 의류 아이템 조회

        Args:
            user_id: 사용자 ID
            clothes_ids: 조회할 clothes_id 리스트
            trace_id: 추적 ID

        Returns:
            조회된 아이템 리스트
        """
        if not clothes_ids:
            return []

        qdrant = await self._get_qdrant()
        log_context = f"trace_id={trace_id} " if trace_id else ""

        logger.info(
            f"Fetching items by IDs | {log_context}"
            f"user_id={user_id} clothes_ids={clothes_ids}"
        )

        must_conditions = [
            qdrant_models.FieldCondition(
                key="userId",
                match=qdrant_models.MatchValue(value=user_id),
            ),
            qdrant_models.FieldCondition(
                key="clothesId",
                match=qdrant_models.MatchAny(any=clothes_ids),
            ),
        ]

        query_filter = qdrant_models.Filter(must=must_conditions)

        response = await qdrant.scroll(
            collection_name=self.settings.qdrant_collection_name,
            scroll_filter=query_filter,
            limit=len(clothes_ids),
            with_payload=True,
        )

        points = response[0]
        candidates = [self._to_candidate_from_record(point) for point in points]

        logger.info(
            f"Fetched items by IDs | {log_context}"
            f"user_id={user_id} requested={len(clothes_ids)} found={len(candidates)}"
        )

        return candidates

    @staticmethod
    def _to_candidate_from_record(record: Record) -> ClothingCandidate:
        payload = record.payload or {}
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
            sub_category=payload.get("subCategory"),
            color=color_list,
            style_tags=payload.get("styleTags", []),
            caption=payload.get("caption"),
            material=payload.get("material") or [],
            season=payload.get("season") or [],
            formality=payload.get("formality") or "",
            occasion=payload.get("occasion") or [],
            similarity_score=0.0,  # scroll은 score 없음
        )

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
            sub_category=payload.get("subCategory"),
            color=color_list,
            style_tags=payload.get("styleTags", []),
            caption=payload.get("caption"),
            material=payload.get("material") or [],
            season=payload.get("season") or [],
            formality=payload.get("formality") or "",
            occasion=payload.get("occasion") or [],
            similarity_score=hit.score,
        )
