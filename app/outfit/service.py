import logging
import uuid
from functools import lru_cache

from langfuse.decorators import observe

from app.common.metrics import OUTFIT_PIPELINE_TOTAL_DURATION, measure_time
from app.outfit.graph import build_outfit_graph
from app.outfit.llm_client import OpenAIClient
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
        self.graph = build_outfit_graph()

    @observe(name="outfit_service.recommend")
    @measure_time(stage="total_pipeline", metric=OUTFIT_PIPELINE_TOTAL_DURATION)
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

        initial_state = {
            "query": request.query,
            "user_id": request.user_id,
            "session_id": request.session_id,
            "trace_id": trace_id,
            "weather": request.weather,
            "upload_slots": request.urls,
        }

        config = {
            "configurable": {
                "query_parser": self.query_parser,
                "search_builder": self.search_builder,
                "clothing_repository": self.repository,
                "outfit_composer": self.composer,
                "vton_processor": self.vton_processor,
            }
        }

        result = await self.graph.ainvoke(initial_state, config=config)
        return result["response"]


# FastAPI 의존성 주입용 (DI 컨테이너 도입 전까지 사용)
@lru_cache
def get_outfit_service() -> OutfitService:
    return OutfitService()
