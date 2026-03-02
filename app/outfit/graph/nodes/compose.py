"""outfit_compose 노드: LLM을 사용해 코디를 구성한다."""

import logging

from langgraph.types import RunnableConfig

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def outfit_compose(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """검색 결과에서 코디를 구성한다."""
    configurable = config.get("configurable", {})
    outfit_composer = configurable["outfit_composer"]

    trace_id = state["trace_id"]
    user_id = state["user_id"]

    parsed_query = state.get("parsed_query")
    if parsed_query is None:
        logger.warning(
            f"Skipping outfit composition due to missing parsed_query | "
            f"trace_id={trace_id}"
        )
        return {
            "response": OutfitResponse(
                query_summary="요청 처리 중 오류가 발생했습니다.",
                outfits=[],
            ),
            "outfits": [],
        }

    search_results = state["search_results"]

    try:
        response = await outfit_composer.compose(
            parsed_query=parsed_query,
            search_results=search_results,
            trace_id=trace_id,
            user_id=user_id,
        )
    except (LLMError, ParseError) as e:
        logger.error(f"Outfit composition failed | trace_id={trace_id}: {e}")
        return {
            "error": str(e),
            "response": OutfitResponse(
                query_summary="코디 구성 중 오류가 발생했습니다.",
                outfits=[],
            ),
            "outfits": [],
        }

    logger.info(
        f"Outfit composition completed | trace_id={trace_id} "
        f"user_id={user_id} outfit_count={len(response.outfits)}"
    )

    return {
        "response": response,
        "outfits": response.outfits,
    }
