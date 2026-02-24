import logging
import uuid
from functools import lru_cache

from app.common.metrics import (
    OUTFIT_PIPELINE_TOTAL_DURATION,
    measure_time,
)
from app.outfit.llm_client import OpenAIClient
from app.shop.query_parser import ShopQueryParser
from app.shop.repository import ShopProductRepository
from app.shop.schemas import ShopSearchRequest, ShopSearchResponse
from app.shop.search_query_builder import ShopSearchQueryBuilder
from app.shop.shop_composer import ShopComposer

logger = logging.getLogger(__name__)


class ShopService:


    def __init__(
        self,
        query_parser: ShopQueryParser | None = None,
        search_builder: ShopSearchQueryBuilder | None = None,
        repository: ShopProductRepository | None = None,
        composer: ShopComposer | None = None,
    ) -> None:
        self.query_parser = query_parser or ShopQueryParser(
            llm_client=OpenAIClient()
        )
        self.search_builder = (
            search_builder or ShopSearchQueryBuilder()
        )
        self.repository = (
            repository or ShopProductRepository()
        )
        self.composer = composer or ShopComposer()

    @measure_time(
        stage="shop_total_pipeline",
        metric=OUTFIT_PIPELINE_TOTAL_DURATION,
    )
    async def search(
        self,
        request: ShopSearchRequest,
        trace_id: str | None = None,
    ) -> ShopSearchResponse:

        if trace_id is None:
            trace_id = str(uuid.uuid4())

        logger.info(
            f"Processing shop search | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"session_id={request.session_id} "
            f'query="{request.query}"'
        )


        parsed = await self.query_parser.parse(
            request.query,
            trace_id=trace_id,
            user_id=request.user_id,
        )
        logger.info(
            f"Parsed shop query | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"occasion={parsed.occasion} "
            f"style={parsed.style} "
            f"price_max={parsed.price_max} "
            f"brand={parsed.brand}"
        )


        search_queries = self.search_builder.build(parsed)
        logger.info(
            f"Generated shop search queries | "
            f"trace_id={trace_id} "
            f"query_count={len(search_queries)}"
        )


        search_results = (
            await self.repository.search_multiple(
                queries=search_queries,
                parsed=parsed,
                trace_id=trace_id,
            )
        )

        total_candidates = sum(
            len(r.candidates) for r in search_results
        )
        logger.info(
            f"Found shop candidates | "
            f"trace_id={trace_id} "
            f"total_candidates={total_candidates}"
        )


        response = await self.composer.compose(
            parsed_query=parsed,
            search_results=search_results,
            trace_id=trace_id,
            user_id=request.user_id,
        )
        response.session_id = request.session_id


        for idx, outfit in enumerate(response.outfits, 1):
            items_str = ", ".join(
                f"{item.title}({item.price:,}원)"
                for item in outfit.items
            )
            logger.info(
                f"Shop outfit composed | "
                f"trace_id={trace_id} "
                f"outfit_index={idx} "
                f"outfit_id={outfit.outfit_id} "
                f"items=[{items_str}]"
            )

        return response



@lru_cache
def get_shop_service() -> ShopService:
    return ShopService()
