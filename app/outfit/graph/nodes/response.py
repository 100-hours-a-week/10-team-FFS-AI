import logging

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def format_response(state: OutfitGraphState) -> dict:
    response = state.get("response")
    session_id = state.get("session_id")
    error = state.get("error")
    shop_supplemented = state.get("shop_supplemented", False)
    fallback_used = state.get("fallback_used", False)

    if error and response is None:
        logger.warning(f"Error in pipeline: {error}")
        response = OutfitResponse(
            query_summary="요청 처리 중 오류가 발생했습니다.",
            outfits=[],
            shop_supplemented=False,
            fallback_used=True,  # 에러 시 fallback으로 간주
        )

    if response and session_id:
        response.session_id = session_id

    # shop_supplemented, fallback_used 설정
    if response:
        response.shop_supplemented = shop_supplemented
        response.fallback_used = fallback_used

    return {"response": response}
