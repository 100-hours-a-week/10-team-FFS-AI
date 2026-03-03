import logging

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import OutfitResponse

logger = logging.getLogger(__name__)


async def format_response(state: OutfitGraphState) -> dict:
    response = state.get("response")
    session_id = state.get("session_id")
    error = state.get("error")

    if error and response is None:
        logger.warning(f"Error in pipeline: {error}")
        response = OutfitResponse(
            query_summary="요청 처리 중 오류가 발생했습니다.",
            outfits=[],
        )

    if response and session_id:
        response.session_id = session_id

    return {"response": response}
