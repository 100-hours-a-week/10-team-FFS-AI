import logging
import time
import uuid
from functools import lru_cache

from app.outfit.llm_client import OpenAIClient
from app.outfit.metrics import PIPELINE_TOTAL_DURATION
from app.outfit.outfit_composer import OutfitComposer
from app.outfit.query_parser import QueryParser
from app.outfit.repository import ClothingRepository
from app.outfit.schemas import OutfitRequest, OutfitResponse
from app.outfit.search_query_builder import SearchQueryBuilder
from app.outfit.vton_processor import VTONProcessor

logger = logging.getLogger(__name__)


class OutfitService:
    def __init__(
        self,
        query_parser: QueryParser | None = None,
        search_builder: SearchQueryBuilder | None = None,
        repository: ClothingRepository | None = None,
        composer: OutfitComposer | None = None,
        vton_processor: VTONProcessor | None = None,
    ) -> None:
        self.query_parser = query_parser or QueryParser(llm_client=OpenAIClient())
        self.search_builder = search_builder or SearchQueryBuilder()
        self.repository = repository or ClothingRepository()
        self.composer = composer or OutfitComposer()
        self.vton_processor = vton_processor or VTONProcessor()

    async def recommend(
        self, request: OutfitRequest, trace_id: str | None = None
    ) -> OutfitResponse:
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        logger.info(
            f"Processing outfit request | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"session_id={request.session_id} "
            f'query="{request.query}"'
        )
        total_start = time.perf_counter()

        parsed = await self.query_parser.parse(
            request.query, trace_id=trace_id, user_id=request.user_id
        )
        logger.info(
            f"Parsed query | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"occasion={parsed.occasion} "
            f"style={parsed.style} "
            f"season={parsed.season} "
            f"formality={parsed.formality}"
        )

        search_queries = self.search_builder.build(parsed)
        logger.info(
            f"Generated search queries | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"query_count={len(search_queries)}"
        )

        search_results = await self.repository.search_multiple(
            user_id=request.user_id,
            queries=search_queries,
            trace_id=trace_id,
        )

        total_candidates = sum(len(r.candidates) for r in search_results)
        logger.info(
            f"Found candidates | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"total_candidates={total_candidates}"
        )

        response = await self.composer.compose(
            parsed_query=parsed,
            search_results=search_results,
            trace_id=trace_id,
            user_id=request.user_id,
        )
        response.session_id = request.session_id

        PIPELINE_TOTAL_DURATION.observe(time.perf_counter() - total_start)

        outfits_detail = []
        for idx, outfit in enumerate(response.outfits, 1):
            items_str = ",".join(str(cid) for cid in outfit.clothes_ids)
            desc_preview = outfit.description[:50] if outfit.description else "N/A"
            outfits_detail.append(
                f"[outfit_{idx}: id={outfit.outfit_id} "
                f"items=[{items_str}] "
                f'desc="{desc_preview}"]'
            )

        logger.info(
            f"Generated outfit recommendations | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"session_id={request.session_id} "
            f"outfit_count={len(response.outfits)} "
            f"outfits={' '.join(outfits_detail)}"
        )

        # VTON 이미지 생성 (urls가 있는 경우에만)
        if request.urls:
            await self.vton_processor.process(response, request.urls)
        else:
            # urls가 없으면 VTON 미요청으로 표시
            for outfit in response.outfits:
                outfit.vton_error = "VTON 미요청 (urls 없음)"

        return response


# FastAPI 의존성 주입용 (DI 컨테이너 도입 전까지 사용)
@lru_cache
def get_outfit_service() -> OutfitService:
    return OutfitService()
