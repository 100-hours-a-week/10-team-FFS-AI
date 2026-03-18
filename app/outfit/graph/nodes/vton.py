"""vton_process 노드: VTON 이미지 생성을 처리한다."""

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def vton_process(state: OutfitGraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    vton_processor = configurable["vton_processor"]

    response = state["response"]
    upload_slots = state.get("upload_slots")
    trace_id = state.get("trace_id", "unknown")

    if upload_slots:
        logger.info(
            f"VTON processing started | trace_id={trace_id} "
            f"upload_slots_count={len(upload_slots)} outfits_count={len(response.outfits)}"
        )
        await vton_processor.process(response, upload_slots)
    else:
        logger.info(
            f"VTON skipped: no upload_slots provided | trace_id={trace_id} "
            f"outfits_count={len(response.outfits)}"
        )
        for outfit in response.outfits:
            outfit.vton_error = "VTON 미요청 (urls 없음)"

    return {"vton_completed": True}
