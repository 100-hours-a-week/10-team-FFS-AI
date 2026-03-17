import logging

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def format_response(state: OutfitGraphState) -> dict:
    response = state.get("response")
    session_id = state.get("session_id")
    error = state.get("error")
    outfit_confidence = state.get("outfit_confidence", 0.8)  # 기본값 0.8

    if error and response is None:
        logger.warning(f"Error in pipeline: {error}")
        response = OutfitResponse(
            query_summary="요청 처리 중 오류가 발생했습니다.",
            outfits=[],
            confidence=0.0,  # 에러 시 confidence 0
        )

    if response and session_id:
        response.session_id = session_id

    # confidence 설정 (아직 없으면)
    if response and not hasattr(response, "confidence"):
        response.confidence = outfit_confidence

    return {"response": response}
