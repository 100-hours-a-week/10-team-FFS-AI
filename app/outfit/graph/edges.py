import logging

from app.outfit.graph.nodes.search import get_min_count
from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)

MAX_SEARCH_RETRIES = 2


def should_research_or_supplement(state: OutfitGraphState) -> str:
    required = state.get("required_categories", [])
    coverage = state.get("category_coverage", {})
    retry = state.get("search_retry_count", 0)

    if not required:
        logger.info("No required_categories → skip to compose (likely error state)")
        return "sufficient"

    total_count = sum(coverage.values())
    if total_count == 0:
        logger.info(
            "Fast path: zero candidates across all categories → "
            "skip retry, go to shop supplement"
        )
        return "supplement_from_shop"

    insufficient = [
        cat for cat in required if coverage.get(cat, 0) < get_min_count(cat)
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
