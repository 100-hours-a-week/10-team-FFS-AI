"""멀티턴 대화 인텐트 판별 노드"""

from __future__ import annotations

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def detect_intent(state: OutfitGraphState, config: RunnableConfig) -> dict:
    # query = state["query"]
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    if not session_data or not session_data.history:
        logger.info(f"Intent detected: new_outfit (no session) | trace_id={trace_id}")
        return {"intent": "new_outfit"}

    # TODO: LLM으로 인텐트 판별 (6단계에서 구현)

    logger.info(f"Intent detected: new_outfit (stub) | trace_id={trace_id}")
    return {"intent": "new_outfit"}
