import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import SearchQuery, SearchResult

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
    # 인텐트 체크: re_request나 style_modify는 검색 스킵
    parsed_intent = state.get("parsed_intent")
    if parsed_intent and parsed_intent["intent_type"] in ["re_request", "style_modify"]:
        trace_id = state.get("trace_id", "unknown")
        logger.info(
            f"Skip vector search for intent: {parsed_intent['intent_type']} | "
            f"trace_id={trace_id}"
        )
        return {"search_results": [], "category_coverage": {}}

    # item_change: 특정 카테고리만 재검색
    if parsed_intent and parsed_intent["intent_type"] == "item_change":
        target_category = parsed_intent.get("target_category")
        if target_category:
            configurable = config.get("configurable", {})
            clothing_repository = configurable["clothing_repository"]

            user_id = state["user_id"]
            trace_id = state.get("trace_id", "unknown")
            search_queries = state.get("search_queries", [])

            logger.info(
                f"Item change: searching for category={target_category} | "
                f"trace_id={trace_id}"
            )

            # 1. confirmed_items에서 고정 아이템 조회
            confirmed_items = state.get("confirmed_items", {})
            confirmed_ids = list(confirmed_items.values())

            confirmed_results = {}
            if confirmed_ids:
                confirmed_candidates = await clothing_repository.get_by_ids(
                    user_id=user_id,
                    clothes_ids=confirmed_ids,
                    trace_id=trace_id,
                )

                # 카테고리별로 그룹화
                for candidate in confirmed_candidates:
                    cat = candidate.category
                    if cat not in confirmed_results:
                        confirmed_results[cat] = []
                    confirmed_results[cat].append(candidate)

                logger.info(
                    f"Loaded confirmed items | trace_id={trace_id} "
                    f"count={len(confirmed_candidates)} "
                    f"categories={list(confirmed_results.keys())}"
                )

            # 2. target_category만 재검색
            search_queries_filtered = [
                q for q in search_queries if q.category_filter == target_category
            ]

            if search_queries_filtered:
                new_search_results = await clothing_repository.search_multiple(
                    user_id=user_id,
                    queries=search_queries_filtered,
                    trace_id=trace_id,
                )
            else:
                new_search_results = []

            # 3. 재검색 결과와 confirmed_items 병합
            all_results = list(new_search_results)
            for cat, candidates in confirmed_results.items():
                if cat != target_category:  # 재검색하지 않은 카테고리만 추가
                    all_results.append(
                        SearchResult(category=cat, candidates=candidates)
                    )

            # 4. category_coverage 계산
            category_coverage = {}
            for result in all_results:
                category_coverage[result.category] = len(result.candidates)

            logger.info(
                f"Item change search completed | trace_id={trace_id} "
                f"target={target_category} coverage={category_coverage}"
            )

            return {
                "search_results": all_results,
                "category_coverage": category_coverage,
            }

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
