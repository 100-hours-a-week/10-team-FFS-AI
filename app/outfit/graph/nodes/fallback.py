import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def build_fallback_response(
    state: OutfitGraphState, config: RunnableConfig
) -> dict:
    outfits = state.get("outfits", [])
    parsed_query = state.get("parsed_query")
    quality_issues = state.get("quality_issues", [])
    error = state.get("error")
    trace_id = state.get("trace_id", "unknown")

    if error:
        fallback_reason = error
    elif quality_issues:
        fallback_reason = f"품질 평가 미달: {', '.join(quality_issues[:3])}"
    else:
        fallback_reason = "추천 품질이 낮을 수 있음"

    for outfit in outfits:
        outfit.fallback_notice = (
            "추천 품질이 기준에 미달하여 재생성되었습니다. 참고용으로 확인해 주세요."
        )

    query_summary = parsed_query.occasion if parsed_query else "코디 추천"

    response = OutfitResponse(
        query_summary=f"{query_summary} (Degraded)",
        outfits=outfits,
        session_id=state.get("session_id"),
    )

    logger.warning(
        f"Fallback response generated | "
        f"trace_id={trace_id} "
        f"reason={fallback_reason} "
        f"outfit_count={len(outfits)}"
    )

    return {
        "response": response,
        "fallback_used": True,
        "fallback_reason": fallback_reason,
    }
