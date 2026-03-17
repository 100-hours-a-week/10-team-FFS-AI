"""세션 데이터 관리 노드"""

from __future__ import annotations

import logging
from datetime import datetime

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import ConversationTurn, SessionData

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


async def save_session_context(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """그래프 실행 결과를 Redis 세션에 저장

    저장 내용:
    - 현재 쿼리와 응답을 conversation_history에 추가
    - 생성된 outfits를 previous_outfits에 저장
    - confirmed_items 업데이트 (향후 구현)

    Args:
        state: 그래프 상태
        config: LangGraph 설정

    Returns:
        빈 dict (state 변경 없음)
    """
    session_id = state.get("session_id")
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    # 세션 ID가 없으면 스킵 (단발성 요청)
    if not session_id:
        logger.info(f"No session_id, skip session save | trace_id={trace_id}")
        return {}

    # SessionManager 가져오기
    session_manager = config["configurable"].get("session_manager")
    if not session_manager:
        logger.warning(
            f"session_manager not in config, skip save | trace_id={trace_id}"
        )
        return {}

    # 세션 데이터 준비
    if session_data is None:
        # 신규 세션 생성
        session_data = SessionData(
            session_id=session_id,
            user_id=state["user_id"],
            history=[],
            previous_outfits=[],
            confirmed_items=[],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

    # 1. 대화 히스토리 업데이트
    query = state["query"]
    response = state.get("response")

    if response:
        # 사용자 쿼리 추가
        session_data.history.append(ConversationTurn(role="user", content=query))

        # 어시스턴트 응답 추가 (query_summary 사용)
        session_data.history.append(
            ConversationTurn(role="assistant", content=response.query_summary)
        )

        # 최대 5턴(10개 메시지) 유지
        if len(session_data.history) > 10:
            session_data.history = session_data.history[-10:]

    # 2. previous_outfits 업데이트
    outfits = state.get("outfits", [])
    if outfits:
        session_data.previous_outfits = outfits  # 최신 추천으로 교체

    # 3. confirmed_items 업데이트 (향후 구현)
    # TODO: item_change 인텐트에서 확정된 아이템 ID를 confirmed_items에 추가

    # 4. Redis에 저장
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
        # 세션 저장 실패는 파이프라인을 막지 않음

    return {}
