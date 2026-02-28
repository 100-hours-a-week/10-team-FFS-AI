"""edges: LangGraph 조건부 엣지 함수들.

Phase 2: should_research_or_supplement
"""

import logging

from app.outfit.graph.nodes.search import get_min_count
from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRIES = 2


def should_research_or_supplement(state: OutfitGraphState) -> str:
    """evaluate_search 이후 다음 경로를 결정한다.

    Returns:
        "sufficient" → outfit_compose
        "retry_search" → relax_and_research → vector_search 루프
        "supplement_from_shop" → 쇼핑 보충 → outfit_compose
    """
    required = state.get("required_categories", [])
    coverage = state.get("category_coverage", {})
    retry = state.get("search_retry_count", 0)

    # fast path: 전 카테고리 0건이면 재검색 스킵
    total_count = sum(coverage.values())
    if total_count == 0:
        logger.info(
            "Fast path: zero candidates across all categories → "
            "skip retry, go to shop supplement"
        )
        return "supplement_from_shop"

    # 부족한 카테고리 확인
    insufficient = [
        cat for cat in required
        if coverage.get(cat, 0) < get_min_count(cat)
    ]

    if not insufficient:
        return "sufficient"
    elif retry < MAX_SEARCH_RETRIES:
        logger.info(
            f"Insufficient categories {insufficient}, "
            f"retry {retry + 1}/{MAX_SEARCH_RETRIES}"
        )
        return "retry_search"
    else:
        logger.info(
            f"Max retries reached ({MAX_SEARCH_RETRIES}), "
            f"still insufficient: {insufficient} → shop supplement"
        )
        return "supplement_from_shop"