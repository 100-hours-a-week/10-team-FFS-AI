import logging
from datetime import datetime

from langgraph.types import RunnableConfig

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import ParsedQuery

logger = logging.getLogger(__name__)

SEASON_KEYWORDS = {
    "봄": ["봄", "spring", "따뜻", "선선"],
    "여름": ["여름", "summer", "더운", "무더위", "시원"],
    "가을": ["가을", "fall", "autumn", "쌀쌀"],
    "겨울": ["겨울", "winter", "추운", "한파", "따뜻하게"],
}
DEFAULT_CATEGORIES = ["TOP", "BOTTOM", "SHOES"]


OUTER_REQUIRED_SEASONS = {"겨울"}
OUTER_OPTIONAL_SEASONS = {"봄", "가을"}

MAX_TPO_RETRIES = 2


async def tpo_extract(state: OutfitGraphState, config: RunnableConfig) -> dict:
    parsed_intent = state.get("parsed_intent")
    if parsed_intent and parsed_intent["intent_type"] != "new":
        trace_id = state.get("trace_id", "unknown")
        logger.info(
            f"Skip TPO extraction for intent: {parsed_intent['intent_type']} | "
            f"trace_id={trace_id}"
        )
        return {}

    configurable = config.get("configurable", {})
    query_parser = configurable["query_parser"]
    search_builder = configurable["search_builder"]
    progress_callback = configurable.get("progress_callback")

    if progress_callback:
        await progress_callback(step="vector_search", step_label="옷장 검색 중...")

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

    if parsed_query.target_category:
        required_categories = [parsed_query.target_category]
        optional_categories: list[str] = []
    else:
        required_categories = list(DEFAULT_CATEGORIES)
        optional_categories = []
        season = parsed_query.season
        if season in OUTER_REQUIRED_SEASONS:
            required_categories.append("OUTER")
        elif season in OUTER_OPTIONAL_SEASONS:
            optional_categories.append("OUTER")

    return {
        "parsed_query": parsed_query,
        "search_queries": search_queries,
        "required_categories": required_categories,
        "optional_categories": optional_categories,
    }


def _extract_season_from_query(query: str) -> str | None:
    query_lower = query.lower()
    for season, keywords in SEASON_KEYWORDS.items():
        if any(kw in query_lower for kw in keywords):
            return season
    return None


def _get_season_from_month() -> str:
    month = datetime.now().month
    if 3 <= month <= 5:
        return "봄"
    elif 6 <= month <= 8:
        return "여름"
    elif 9 <= month <= 11:
        return "가을"
    else:
        return "겨울"


async def tpo_validate(state: OutfitGraphState, config: RunnableConfig) -> dict:
    trace_id = state.get("trace_id", "unknown")

    if state.get("error"):
        logger.warning(f"TPO validation failed: error exists | trace_id={trace_id}")
        return {"quality_passed": False}

    parsed_query = state.get("parsed_query")
    if not parsed_query:
        logger.warning(f"TPO validation failed: no parsed_query | trace_id={trace_id}")
        return {"quality_passed": False}

    if not parsed_query.occasion or not parsed_query.occasion.strip():
        logger.warning(f"TPO validation failed: empty occasion | trace_id={trace_id}")
        return {"quality_passed": False}

    if not parsed_query.style or not parsed_query.style.strip():
        logger.warning(f"TPO validation failed: empty style | trace_id={trace_id}")
        return {"quality_passed": False}

    query = state.get("query", "")
    query_season = _extract_season_from_query(query)
    if query_season and not parsed_query.season:
        logger.warning(
            f"TPO validation failed: season keyword in query but not parsed | "
            f"trace_id={trace_id} query_season={query_season}"
        )
        return {"quality_passed": False}

    logger.info(f"TPO validation passed | trace_id={trace_id}")
    return {"quality_passed": True}


async def tpo_retry(state: OutfitGraphState, config: RunnableConfig) -> dict:
    trace_id = state.get("trace_id", "unknown")
    current_retry = state.get("tpo_retry_count", 0)

    logger.info(
        f"TPO retry attempt | trace_id={trace_id} retry_count={current_retry + 1}"
    )

    return {
        "tpo_retry_count": current_retry + 1,
        "error": None,
        "quality_passed": False,
    }


async def tpo_fallback(state: OutfitGraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    search_builder = configurable["search_builder"]

    trace_id = state.get("trace_id", "unknown")
    query = state.get("query", "")

    query_season = _extract_season_from_query(query)
    season = query_season or _get_season_from_month()

    parsed_query = ParsedQuery(
        occasion="일상",
        style="깔끔한",
        season=season,
        formality=None,
    )

    logger.warning(
        f"TPO fallback used | trace_id={trace_id} "
        f"occasion={parsed_query.occasion} style={parsed_query.style} "
        f"season={parsed_query.season}"
    )

    search_queries = search_builder.build(parsed_query)

    required_categories = list(DEFAULT_CATEGORIES)
    optional_categories: list[str] = []
    if season in OUTER_REQUIRED_SEASONS:
        required_categories.append("OUTER")
    elif season in OUTER_OPTIONAL_SEASONS:
        optional_categories.append("OUTER")

    return {
        "parsed_query": parsed_query,
        "search_queries": search_queries,
        "required_categories": required_categories,
        "optional_categories": optional_categories,
        "tpo_fallback_used": True,
        "error": None,
        "quality_passed": True,
    }
