"""멀티턴 대화 세션 관리"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from redis.asyncio import Redis

from app.core.database import get_redis_client
from app.outfit.schemas import Message, Outfit, SessionData

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 1800


class SessionManager:
    def __init__(self, redis_client: Redis | None = None) -> None:
        self.redis = redis_client or get_redis_client()

    def _session_key(self, session_id: str) -> str:
        return f"session:{session_id}"

    async def load_session(self, session_id: str) -> SessionData | None:
        key = self._session_key(session_id)
        data = await self.redis.hgetall(key)

        if not data:
            logger.info(f"Session not found: session_id={session_id}")
            return None

        session_data = SessionData(
            session_id=session_id,
            user_id=int(data["user_id"]),
            history=[Message(**turn) for turn in json.loads(data.get("history", "[]"))],
            previous_outfits=[
                Outfit(**outfit)
                for outfit in json.loads(data.get("previous_outfits", "[]"))
            ],
            confirmed_items=json.loads(data.get("confirmed_items", "{}")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

        logger.info(
            f"Session loaded: session_id={session_id}, "
            f"user_id={session_data.user_id}, "
            f"history_turns={len(session_data.history)}, "
            f"previous_outfits={len(session_data.previous_outfits)}"
        )

        return session_data

    async def save_session(self, session_data: SessionData) -> None:
        key = self._session_key(session_data.session_id)

        session_data.updated_at = datetime.utcnow()

        data = {
            "user_id": str(session_data.user_id),
            "history": json.dumps(
                [turn.model_dump() for turn in session_data.history],
                default=str,
            ),
            "previous_outfits": json.dumps(
                [outfit.model_dump() for outfit in session_data.previous_outfits],
                default=str,
            ),
            "confirmed_items": json.dumps(session_data.confirmed_items),
            "created_at": session_data.created_at.isoformat(),
            "updated_at": session_data.updated_at.isoformat(),
        }

        await self.redis.hset(key, mapping=data)
        await self.redis.expire(key, SESSION_TTL_SECONDS)

        logger.info(
            f"Session saved: session_id={session_data.session_id}, "
            f"ttl={SESSION_TTL_SECONDS}s"
        )

    async def append_turn(
        self,
        session_id: str,
        user_id: int,
        user_message: str,
        assistant_message: str,
    ) -> None:
        session_data = await self.load_session(session_id)

        if session_data is None:
            session_data = SessionData(
                session_id=session_id,
                user_id=user_id,
            )

        session_data.history.append(
            Message(role="user", content=user_message, outfits=None)
        )
        session_data.history.append(
            Message(role="assistant", content=assistant_message, outfits=None)
        )

        # 최대 5턴(10개 메시지)만 유지
        if len(session_data.history) > 10:
            session_data.history = session_data.history[-10:]

        await self.save_session(session_data)

    async def delete_session(self, session_id: str) -> None:
        key = self._session_key(session_id)
        await self.redis.delete(key)
        logger.info(f"Session deleted: session_id={session_id}")
