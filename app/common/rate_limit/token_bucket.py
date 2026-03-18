"""
Redis 기반 Token Bucket Rate Limiter
====================================
분산 환경에서 여러 Worker 인스턴스가 공유하는 토큰 버킷.
LLM API Rate Limit 사전 방지를 위해 사용.

토큰 버킷 알고리즘:
- capacity: 버킷의 최대 토큰 수 (버스트 허용량)
- rate: 분당 토큰 생성 속도 (RPM)
- refill_per_second: 초당 토큰 리필 속도
- 토큰이 부족하면 리필될 때까지 대기
"""

import asyncio
import logging
import time

from redis.asyncio import Redis

try:
    from app.workers.metrics import RATE_LIMIT_HITS_TOTAL, RATE_LIMIT_WAIT_SECONDS

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False

logger = logging.getLogger(__name__)


class TokenBucket:
    """Redis 기반 분산 토큰 버킷 Rate Limiter"""

    def __init__(
        self,
        redis_client: Redis,
        key: str,
        rate: int = 60,
        capacity: int | None = None,
    ) -> None:
        """TokenBucket 초기화

        Args:
            redis_client: Redis 비동기 클라이언트
            key: Redis 키 이름 (자동으로 rate_limit: 접두사 추가)
            rate: 분당 토큰 생성 속도 (RPM, 기본: 60)
            capacity: 버킷 최대 용량 (기본: rate와 동일)
        """
        self.redis = redis_client
        self.key = f"rate_limit:{key}"
        self.rate = rate
        self.capacity = capacity or rate
        self.refill_per_second = rate / 60.0

        logger.info(
            f"TokenBucket initialized | "
            f"key={self.key} rate={self.rate}RPM "
            f"capacity={self.capacity} "
            f"refill_per_second={self.refill_per_second:.2f}"
        )

    async def acquire(self) -> bool:
        """토큰 획득 (없으면 대기)

        Returns:
            bool: 항상 True (획득 성공)

        Note:
            토큰이 부족하면 리필될 때까지 자동으로 대기합니다.
        """
        while True:
            current_time = time.time()

            # Lua 스크립트로 atomic 연산
            tokens = await self._check_and_consume(current_time)

            if tokens >= 1.0:
                logger.debug(
                    f"Token acquired | key={self.key} remaining={tokens-1:.2f}"
                )
                return True

            # 토큰 부족, 대기
            wait_time = (1.0 - tokens) / self.refill_per_second
            logger.warning(
                f"Rate limit reached | key={self.key} waiting={wait_time:.2f}s"
            )

            # Prometheus 메트릭 기록 (optional)
            if _METRICS_AVAILABLE:
                RATE_LIMIT_HITS_TOTAL.labels(key=self.key).inc()
                RATE_LIMIT_WAIT_SECONDS.labels(key=self.key).observe(wait_time)

            await asyncio.sleep(wait_time)

    async def _check_and_consume(self, current_time: float) -> float:
        """Redis에서 토큰 확인 및 소비 (Lua 스크립트)

        Args:
            current_time: 현재 시각 (Unix timestamp)

        Returns:
            float: 소비 후 남은 토큰 수
        """
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])

        -- 현재 버킷 상태 조회
        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time

        -- 경과 시간만큼 토큰 리필
        local elapsed = current_time - last_refill
        tokens = math.min(capacity, tokens + elapsed * refill_rate)

        -- 토큰 소비 (1개)
        if tokens >= 1.0 then
            tokens = tokens - 1.0
        end

        -- Redis 업데이트 (tokens, last_refill)
        redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
        redis.call('EXPIRE', key, 120)

        return tokens
        """

        result = await self.redis.eval(
            lua_script,
            1,
            self.key,
            str(self.capacity),
            str(self.refill_per_second),
            str(current_time),
        )
        return float(result)

    async def get_remaining_tokens(self) -> float:
        """현재 남은 토큰 수 조회 (소비하지 않음)

        Returns:
            float: 남은 토큰 수
        """
        current_time = time.time()

        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local current_time = tonumber(ARGV[3])

        local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(bucket[1]) or capacity
        local last_refill = tonumber(bucket[2]) or current_time

        local elapsed = current_time - last_refill
        tokens = math.min(capacity, tokens + elapsed * refill_rate)

        return tokens
        """

        result = await self.redis.eval(
            lua_script,
            1,
            self.key,
            str(self.capacity),
            str(self.refill_per_second),
            str(current_time),
        )
        return float(result)
