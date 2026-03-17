from __future__ import annotations

import logging
from datetime import datetime

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import Message, SessionData

logger = logging.getLogger(__name__)


async def load_session_context(state: OutfitGraphState, config: RunnableConfig) -> dict:
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    if not session_data:
        logger.info(f"No session context to load | trace_id={trace_id}")
        return {}

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


async def save_session_context(state: OutfitGraphState, config: RunnableConfig) -> dict:
    session_id = state.get("session_id")
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    if not session_id:
        logger.info(f"No session_id, skip session save | trace_id={trace_id}")
        return {}

    session_manager = config["configurable"].get("session_manager")
    if not session_manager:
        logger.warning(
            f"session_manager not in config, skip save | trace_id={trace_id}"
        )
        return {}

    if session_data is None:
        session_data = SessionData(
            session_id=session_id,
            user_id=state["user_id"],
            history=[],
            previous_outfits=[],
            confirmed_items=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    query = state["query"]
    response = state.get("response")
    outfits = state.get("outfits", [])

    if response:
        session_data.history.append(Message(role="user", content=query, outfits=None))

        session_data.history.append(
            Message(
                role="assistant",
                content=response.query_summary,
                outfits=outfits if outfits else None,
            )
        )

        if len(session_data.history) > 10:
            session_data.history = session_data.history[-10:]
    if outfits:
        session_data.previous_outfits = outfits

    # 3. confirmed_items 업데이트 (향후 구현)
    # TODO: item_change 인텐트에서 확정된 아이템 ID를 confirmed_items에 추가

    try:
        await session_manager.save_session(session_data)
        logger.info(
            f"Session saved | session_id={session_id} | "
            f"history_turns={len(session_data.history)} | "
            f"previous_outfits={len(session_data.previous_outfits)} | "
            f"trace_id={trace_id}"
        )
    except Exception as e:
        logger.exception(
            f"Failed to save session | session_id={session_id} | "
            f"trace_id={trace_id} | error={e}"
        )

    return {}
