"""세션 데이터 관리 노드"""

from __future__ import annotations

import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def load_session_context(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """세션 데이터에서 컨텍스트를 추출하여 state에 주입한다.

    Args:
        state: 그래프 상태
        config: LangGraph 설정

    Returns:
        dict: 세션 컨텍스트 (previous_outfits, confirmed_items)
    """
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    if not session_data:
        logger.info(f"No session context to load | trace_id={trace_id}")
        return {}

    # previous_outfits를 merged_candidates에 추가 (멀티턴 추천 시 참고)
    previous_outfits = session_data.previous_outfits
    confirmed_items = session_data.confirmed_items

    logger.info(
        f"Session context loaded | trace_id={trace_id} "
        f"previous_outfits={len(previous_outfits)} "
        f"confirmed_items={confirmed_items}"
    )

    # TODO: 6단계에서 실제 컨텍스트 주입 로직 구현
    return {
        "previous_outfits": previous_outfits,
        "confirmed_items": confirmed_items,
    }
