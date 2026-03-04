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


MAX_QUALITY_RETRIES = 1


def should_retry_or_fallback(state: OutfitGraphState) -> str:
    """품질 평가 후 다음 경로를 결정한다.

    Returns:
        "pass": 품질 통과 → vton_process로 이동
        "retry_compose": 재시도 가능 → 조합 서브그래프 재실행
        "fallback": 재시도 불가 → build_fallback_response로 이동
    """
    quality_passed = state.get("quality_passed", False)
    quality_retry_count = state.get("quality_retry_count", 0)
    trace_id = state.get("trace_id", "unknown")

    if quality_passed:
        logger.info(f"Quality passed → vton_process | trace_id={trace_id}")
        return "pass"
    elif quality_retry_count <= MAX_QUALITY_RETRIES:
        logger.info(
            f"Quality failed, retry {quality_retry_count}/{MAX_QUALITY_RETRIES} "
            f"→ retry compose | trace_id={trace_id}"
        )
        return "retry_compose"
    else:
        logger.warning(
            f"Quality failed after {MAX_QUALITY_RETRIES} retries "
            f"→ fallback | trace_id={trace_id}"
        )
        return "fallback"
