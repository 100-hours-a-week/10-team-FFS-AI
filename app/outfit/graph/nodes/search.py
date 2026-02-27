"""vector_search 노드: Qdrant 벡터 검색을 수행한다."""

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def vector_search(
    state: OutfitGraphState, config: RunnableConfig
) -> dict:
    """검색 쿼리로 Qdrant에서 의류 후보를 검색한다."""
    configurable = config.get("configurable", {})
    clothing_repository = configurable["clothing_repository"]

    user_id = state["user_id"]
    trace_id = state["trace_id"]

    search_queries = state.get("search_queries")
    if not search_queries:
        logger.warning(
            f"Skipping vector search due to missing search_queries | "
            f"trace_id={trace_id}"
        )
        return {"search_results": [], "category_coverage": {}}

    search_results = await clothing_repository.search_multiple(
        user_id=user_id,
        queries=search_queries,
        trace_id=trace_id,
    )

    # category_coverage 계산
    category_coverage: dict[str, int] = {}
    for result in search_results:
        category_coverage[result.category] = len(result.candidates)

    total_candidates = sum(category_coverage.values())
    logger.info(
        f"Found candidates | trace_id={trace_id} "
        f"user_id={user_id} total_candidates={total_candidates} "
        f"coverage={category_coverage}"
    )

    return {
        "search_results": search_results,
        "category_coverage": category_coverage,
    }
