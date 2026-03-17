import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from functools import lru_cache

from app.common.metrics import OUTFIT_PIPELINE_TOTAL_DURATION, measure_time
from app.outfit.graph import build_outfit_graph
from app.outfit.llm_client import OpenAIClient
from app.outfit.outfit_composer import OutfitComposer
from app.outfit.query_parser import QueryParser
from app.outfit.repository import ClothingRepository
from app.outfit.schemas import OutfitRequest, OutfitResponse, SessionData
from app.outfit.search_query_builder import SearchQueryBuilder
from app.outfit.vton_processor import VTONProcessor
from app.shop.repository import ShopProductRepository
from app.shop.search_query_builder import ShopSearchQueryBuilder

logger = logging.getLogger(__name__)


PIPELINE_TIMEOUT_SECONDS = 90

# Progress 발행 콜백 타입
ProgressCallback = Callable[[int, str], Awaitable[None]]


class OutfitService:
    def __init__(
        self,
        query_parser: QueryParser | None = None,
        search_builder: SearchQueryBuilder | None = None,
        repository: ClothingRepository | None = None,
        composer: OutfitComposer | None = None,
        vton_processor: VTONProcessor | None = None,
        shop_repository: ShopProductRepository | None = None,
        shop_search_builder: ShopSearchQueryBuilder | None = None,
    ) -> None:
        self.query_parser = query_parser or QueryParser(llm_client=OpenAIClient())
        self.search_builder = search_builder or SearchQueryBuilder()
        self.repository = repository or ClothingRepository()
        self.composer = composer or OutfitComposer()
        self.vton_processor = vton_processor or VTONProcessor()
        self.shop_repository = shop_repository or ShopProductRepository()
        self.shop_search_builder = shop_search_builder or ShopSearchQueryBuilder()
        self.graph = build_outfit_graph()

    @measure_time(stage="total_pipeline", metric=OUTFIT_PIPELINE_TOTAL_DURATION)
    async def recommend(
        self,
        request: OutfitRequest,
        trace_id: str | None = None,
        session_data: SessionData | None = None,
        progress_callback: ProgressCallback | None = None,
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

        initial_state = {
            "query": request.query,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "trace_id": trace_id,
            "weather": request.weather,
            "upload_slots": request.urls,
            "quality_retry_count": 0,
            "session_data": session_data,  # 세션 데이터 추가
        }
        from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

        langfuse_handler = LangfuseCallbackHandler()

        config = {
            "configurable": {
                "query_parser": self.query_parser,
                "search_builder": self.search_builder,
                "clothing_repository": self.repository,
                "outfit_composer": self.composer,
                "vton_processor": self.vton_processor,
                "shop_repository": self.shop_repository,
                "shop_search_builder": self.shop_search_builder,
                "progress_callback": progress_callback,  # Progress 콜백 추가
            },
            "callbacks": [langfuse_handler],
            "metadata": {
                "langfuse_user_id": str(request.user_id),
                "langfuse_session_id": request.session_id,
                "langfuse_tags": ["langgraph", "outfit"],
            },
        }

        try:
            result = await asyncio.wait_for(
                self.graph.ainvoke(initial_state, config=config),
                timeout=PIPELINE_TIMEOUT_SECONDS,
            )
            return result["response"]
        except TimeoutError:
            logger.error(
                f"Pipeline timeout after {PIPELINE_TIMEOUT_SECONDS}s | "
                f"trace_id={trace_id} user_id={request.user_id}"
            )
            return OutfitResponse(
                query_summary="코디 추천 (시간 초과)",
                outfits=[],
                session_id=request.session_id,
            )


# FastAPI 의존성 주입용 (DI 컨테이너 도입 전까지 사용)
@lru_cache
def get_outfit_service() -> OutfitService:
    return OutfitService()
