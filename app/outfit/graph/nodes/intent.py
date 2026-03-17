"""멀티턴 대화 인텐트 판별 노드"""

from __future__ import annotations

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState, ParsedIntent

logger = logging.getLogger(__name__)


async def detect_intent(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """사용자 의도를 분석하여 ParsedIntent 반환 (V3)"""
    # query = state["query"]
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    # TODO: LLM 호출로 ParsedIntent 생성
    # 임시로 기본값 반환 (새로운 코디 추천 요청으로 간주)
    default_intent: ParsedIntent = {
        "intent_type": "new",
        "target_outfit_index": None,
        "target_category": None,
        "style_direction": None,
    }

    if not session_data or not session_data.history:
        logger.info(f"Intent detected: new (no session) | trace_id={trace_id}")
        return {"intent": "new_outfit", "parsed_intent": default_intent}

    # 세션이 있어도 현재는 기본 인텐트 반환 (LLM 구현 전)
    logger.info(f"Intent detected: new (stub) | trace_id={trace_id}")
    return {"intent": "new_outfit", "parsed_intent": default_intent}
