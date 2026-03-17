"""SessionManager 테스트"""

import pytest

from app.outfit.schemas import Message, SessionData
from app.outfit.session_manager import SessionManager


@pytest.mark.asyncio
async def test_session_save_and_load(redis_client) -> None:
    """세션 저장 및 로드 테스트"""
    manager = SessionManager(redis_client)

    session_data = SessionData(
        session_id="sess-001",
        user_id=1,
        history=[
            Message(role="user", content="안녕", outfits=None),
            Message(role="assistant", content="안녕하세요", outfits=None),
        ],
        previous_outfits=[],
        confirmed_items=[],
    )

    await manager.save_session(session_data)

    loaded = await manager.load_session("sess-001")

    assert loaded is not None
    assert loaded.user_id == 1
    assert len(loaded.history) == 2
    assert loaded.history[0].role == "user"
    assert loaded.history[0].content == "안녕"
    assert loaded.history[1].role == "assistant"
    assert loaded.history[1].content == "안녕하세요"


@pytest.mark.asyncio
async def test_session_append_turn(redis_client) -> None:
    """대화 턴 추가 테스트"""
    manager = SessionManager(redis_client)

    await manager.append_turn(
        session_id="sess-002",
        user_id=2,
        user_message="데이트 코디 추천해줘",
        assistant_message="깔끔한 데이트 코디 추천드립니다.",
    )

    loaded = await manager.load_session("sess-002")

    assert loaded is not None
    assert len(loaded.history) == 2
    assert loaded.history[0].content == "데이트 코디 추천해줘"
    assert loaded.history[1].content == "깔끔한 데이트 코디 추천드립니다."


@pytest.mark.asyncio
async def test_session_ttl(redis_client) -> None:
    """세션 TTL 확인"""
    manager = SessionManager(redis_client)

    session_data = SessionData(
        session_id="sess-ttl-001",
        user_id=1,
    )

    await manager.save_session(session_data)

    # TTL 확인
    ttl = await redis_client.ttl("session:sess-ttl-001")
    assert ttl > 0
    assert ttl <= 1800


@pytest.mark.asyncio
async def test_session_not_found(redis_client) -> None:
    """존재하지 않는 세션 로드 테스트"""
    manager = SessionManager(redis_client)

    loaded = await manager.load_session("sess-not-exist")

    assert loaded is None


@pytest.mark.asyncio
async def test_session_delete(redis_client) -> None:
    """세션 삭제 테스트"""
    manager = SessionManager(redis_client)

    session_data = SessionData(
        session_id="sess-del-001",
        user_id=1,
    )

    await manager.save_session(session_data)

    # 삭제 전 확인
    loaded = await manager.load_session("sess-del-001")
    assert loaded is not None

    # 삭제
    await manager.delete_session("sess-del-001")

    # 삭제 후 확인
    loaded = await manager.load_session("sess-del-001")
    assert loaded is None


@pytest.mark.asyncio
async def test_session_history_limit(redis_client) -> None:
    """대화 히스토리 최대 10개 제한 테스트"""
    manager = SessionManager(redis_client)

    # 11턴 추가 (22개 메시지)
    for i in range(11):
        await manager.append_turn(
            session_id="sess-limit-001",
            user_id=1,
            user_message=f"질문 {i}",
            assistant_message=f"답변 {i}",
        )

    loaded = await manager.load_session("sess-limit-001")

    assert loaded is not None
    # 최대 10개 메시지만 유지
    assert len(loaded.history) == 10
    # 가장 오래된 턴이 제거되고 최근 10개만 유지 (질문 6부터)
    assert loaded.history[0].content == "질문 6"
    assert loaded.history[-1].content == "답변 10"
