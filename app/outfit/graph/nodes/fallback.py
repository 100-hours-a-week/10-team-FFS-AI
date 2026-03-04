"""Fallback 응답 생성 노드.

품질 평가 미달 또는 에러 시 Degraded 응답을 생성한다.
"""

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def build_fallback_response(
    state: OutfitGraphState, config: RunnableConfig
) -> dict:
    """품질 평가 미달 또는 에러 시 Degraded 응답을 생성한다.

    읽는 State 필드: outfits, parsed_query, fallback_reason, quality_issues, error
    쓰는 State 필드: response, fallback_used, fallback_reason
    """
    outfits = state.get("outfits", [])
    parsed_query = state.get("parsed_query")
    quality_issues = state.get("quality_issues", [])
    error = state.get("error")
    trace_id = state.get("trace_id", "unknown")

    # fallback_reason 결정
    if error:
        fallback_reason = error
    elif quality_issues:
        # 최대 3개만 표시
        fallback_reason = f"품질 평가 미달: {', '.join(quality_issues[:3])}"
    else:
        fallback_reason = "추천 품질이 낮을 수 있음"

    # 각 outfit에 fallback_notice 설정
    for outfit in outfits:
        outfit.fallback_notice = (
            "추천 품질이 기준에 미달하여 재생성되었습니다. 참고용으로 확인해 주세요."
        )

    # query_summary 생성
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
