"""tpo_extract 노드: 사용자 쿼리를 파싱하고 검색 쿼리를 생성한다."""

import logging

from langgraph.types import RunnableConfig

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)

WINTER_KEYWORDS = ["겨울", "winter", "추운", "한파"]
DEFAULT_CATEGORIES = ["TOP", "BOTTOM", "SHOES"]


async def tpo_extract(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """사용자 쿼리를 ParsedQuery로 파싱하고 SearchQuery 목록을 생성한다."""
    configurable = config.get("configurable", {})
    query_parser = configurable["query_parser"]
    search_builder = configurable["search_builder"]

    query = state["query"]
    user_id = state["user_id"]
    trace_id = state["trace_id"]

    try:
        parsed_query = await query_parser.parse(
            query, trace_id=trace_id, user_id=user_id
        )
    except (LLMError, ParseError) as e:
        logger.error(f"TPO extraction failed | trace_id={trace_id}: {e}")
        return {"error": str(e)}

    logger.info(
        f"Parsed query | trace_id={trace_id} user_id={user_id} "
        f"occasion={parsed_query.occasion} style={parsed_query.style} "
        f"season={parsed_query.season} formality={parsed_query.formality}"
    )

    search_queries = search_builder.build(parsed_query)
    logger.info(
        f"Generated search queries | trace_id={trace_id} "
        f"user_id={user_id} query_count={len(search_queries)}"
    )

    # required_categories 결정
    if parsed_query.target_category:
        required_categories = [parsed_query.target_category]
    else:
        required_categories = list(DEFAULT_CATEGORIES)
        if parsed_query.season and any(
            kw in parsed_query.season.lower() for kw in WINTER_KEYWORDS
        ):
            required_categories.append("OUTER")

    return {
        "parsed_query": parsed_query,
        "search_queries": search_queries,
        "required_categories": required_categories,
    }
