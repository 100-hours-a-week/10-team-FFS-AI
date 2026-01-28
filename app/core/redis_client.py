"""
Redis 클라이언트 (Singleton)

배치 작업 상태 관리용 Redis 클라이언트
"""

from __future__ import annotations

import json
import logging

import redis

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RedisClient:
    """Redis 클라이언트 래퍼"""

    def __init__(self: RedisClient) -> None:
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            password=settings.redis_password,
            decode_responses=True,  # 자동으로 bytes를 str로 변환
            max_connections=settings.redis_max_connections,
        )

    def set_batch(
        self: RedisClient, batch_id: str, data: dict[str, object], ttl: int = 86400
    ) -> None:
        """
        배치 정보 저장

        Args:
            batch_id: 배치 ID
            data: 배치 데이터
            ttl: 만료 시간 (초, 기본 24시간)
        """
        key = f"batch:{batch_id}"
        self.client.setex(key, ttl, json.dumps(data))
        logger.debug(f"Redis SET: {key}, TTL: {ttl}s")

    def get_batch(self: RedisClient, batch_id: str) -> dict[str, object] | None:
        """
        배치 정보 조회

        Args:
            batch_id: 배치 ID

        Returns:
            배치 데이터 또는 None
        """
        key = f"batch:{batch_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def exists_batch(self: RedisClient, batch_id: str) -> bool:
        """
        배치 존재 여부 확인

        Args:
            batch_id: 배치 ID

        Returns:
            존재 여부
        """
        key = f"batch:{batch_id}"
        return self.client.exists(key) > 0

    def set_task(
        self: RedisClient, task_id: str, data: dict[str, object], ttl: int = 86400
    ) -> None:
        """
        작업 정보 저장

        Args:
            task_id: 작업 ID
            data: 작업 데이터
            ttl: 만료 시간 (초, 기본 24시간)
        """
        key = f"task:{task_id}"
        self.client.setex(key, ttl, json.dumps(data))
        logger.debug(f"Redis SET: {key}, TTL: {ttl}s")

    def get_task(self: RedisClient, task_id: str) -> dict[str, object] | None:
        """
        작업 정보 조회

        Args:
            task_id: 작업 ID

        Returns:
            작업 데이터 또는 None
        """
        key = f"task:{task_id}"
        data = self.client.get(key)
        if data:
            return json.loads(data)
        return None

    def get_tasks_by_batch(self: RedisClient, batch_id: str) -> list[dict[str, object]]:
        """
        배치에 속한 모든 작업 조회

        Args:
            batch_id: 배치 ID

        Returns:
            작업 목록
        """
        # 배치에 속한 task_id 목록을 저장하는 Set 사용
        set_key = f"batch:{batch_id}:tasks"
        task_ids = self.client.smembers(set_key)

        tasks = []
        for task_id in task_ids:
            task = self.get_task(task_id)
            if task:
                tasks.append(task)

        return tasks

    def add_task_to_batch(
        self: RedisClient, batch_id: str, task_id: str, ttl: int = 86400
    ) -> None:
        """
        배치에 작업 추가 (Set에 저장)

        Args:
            batch_id: 배치 ID
            task_id: 작업 ID
            ttl: 만료 시간 (초)
        """
        set_key = f"batch:{batch_id}:tasks"
        self.client.sadd(set_key, task_id)
        self.client.expire(set_key, ttl)

    def update_batch_field(
        self: RedisClient, batch_id: str, field: str, value: object
    ) -> None:
        """
        배치의 특정 필드 업데이트

        Args:
            batch_id: 배치 ID
            field: 필드명
            value: 값
        """
        batch = self.get_batch(batch_id)
        if batch:
            batch[field] = value
            self.set_batch(batch_id, batch)

    def increment_batch_completed(self: RedisClient, batch_id: str) -> None:
        """
        배치의 완료 카운트 증가

        Args:
            batch_id: 배치 ID
        """
        batch = self.get_batch(batch_id)
        if batch:
            batch["completed"] = batch.get("completed", 0) + 1
            batch["processing"] = max(0, batch.get("processing", 0) - 1)
            self.set_batch(batch_id, batch)

    def ping(self: RedisClient) -> bool:
        """
        Redis 연결 확인

        Returns:
            연결 성공 여부
        """
        try:
            return self.client.ping()
        except Exception as e:
            logger.error(f"Redis ping 실패: {e}")
            return False


# 싱글톤 인스턴스
_redis_client: RedisClient | None = None


def get_redis_client() -> RedisClient:
    """Redis 클라이언트 싱글톤 가져오기"""
    global _redis_client
    if _redis_client is None:
        _redis_client = RedisClient()
    return _redis_client
