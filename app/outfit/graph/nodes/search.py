import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import SearchQuery

logger = logging.getLogger(__name__)


MIN_CANDIDATE_COUNTS: dict[str, int] = {
    "TOP": 2,
    "BOTTOM": 2,
    "OUTER": 1,
    "SHOES": 1,
    "DRESS": 0,
    "ACCESSORY": 0,
    "ETC": 0,
}


def get_min_count(category: str) -> int:
    return MIN_CANDIDATE_COUNTS.get(category, 0)


async def vector_search(state: OutfitGraphState, config: RunnableConfig) -> dict:
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


async def evaluate_search(state: OutfitGraphState) -> dict:
    search_results = state.get("search_results", [])
    required_categories = state.get("required_categories", [])
    # optional_categories는 insufficient 판정 제외
    # — coverage에 있으면 merged_candidates에 자동 포함되어 코디에 활용됨
    optional_categories = state.get("optional_categories", [])

    category_coverage: dict[str, int] = {}
    for result in search_results:
        category_coverage[result.category] = len(result.candidates)

    # required만 insufficient 체크 (optional은 부족해도 코디 성립)
    insufficient = [
        cat
        for cat in required_categories
        if category_coverage.get(cat, 0) < get_min_count(cat)
    ]

    if optional_categories:
        logger.debug(
            f"Optional categories skipped in insufficient check: {optional_categories}"
        )

    total_count = sum(category_coverage.values())

    if not insufficient:
        logger.info(f"Search sufficient | coverage={category_coverage}")
        return {
            "category_coverage": category_coverage,
            "merged_candidates": list(search_results),
        }
    else:
        logger.info(
            f"Search insufficient | "
            f"total={total_count} coverage={category_coverage} "
            f"missing={insufficient}"
        )
        return {
            "category_coverage": category_coverage,
        }


async def relax_and_research(state: OutfitGraphState) -> dict:
    search_queries = state.get("search_queries", [])
    filter_level = state.get("filter_level", "strict")
    search_retry_count = state.get("search_retry_count", 0)

    if filter_level == "strict":
        new_queries = [
            SearchQuery(
                text=q.category_filter or "",
                category_filter=q.category_filter,
            )
            for q in search_queries
        ]
        new_level = "relaxed"
    else:
        new_queries = [
            SearchQuery(text="", category_filter=None) for _ in search_queries
        ]
        new_level = "minimal"

    logger.info(
        f"Relaxing search filters | "
        f"{filter_level} → {new_level} "
        f"retry={search_retry_count + 1}"
    )

    return {
        "search_queries": new_queries,
        "filter_level": new_level,
        "search_retry_count": search_retry_count + 1,
    }
