"""vton_process 노드: VTON 이미지 생성을 처리한다."""

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def vton_process(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """VTON 이미지를 생성하고 S3에 업로드한다."""
    configurable = config.get("configurable", {})
    vton_processor = configurable["vton_processor"]

    response = state["response"]
    upload_slots = state.get("upload_slots", [])

    if upload_slots:
        await vton_processor.process(response, upload_slots)
    else:
        for outfit in response.outfits:
            outfit.vton_error = "VTON 미요청 (urls 없음)"

    return {"vton_completed": True}
