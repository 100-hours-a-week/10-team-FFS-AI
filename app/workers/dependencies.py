import logging

from app.common.rate_limit import TokenBucket
from app.config import get_settings
from app.core.database import close_databases, get_redis_client, init_databases
from app.outfit.llm_client import OpenAIClient
from app.outfit.service import OutfitService

logger = logging.getLogger(__name__)

_outfit_service: OutfitService | None = None


async def init_worker_dependencies() -> None:
    global _outfit_service

    logger.info("Worker 의존성 초기화 시작...")

    await init_databases()

    settings = get_settings()
    redis_client = get_redis_client()
    rate_limiter = TokenBucket(
        redis_client=redis_client,
        key="openai_api",
        rate=settings.openai_rate_limit_rpm or 60,
    )
    logger.info(
        f"Rate limiter created | rate={settings.openai_rate_limit_rpm or 60}RPM"
    )

    llm_client = OpenAIClient(rate_limiter=rate_limiter)

    _outfit_service = OutfitService(llm_client=llm_client)

    logger.info("Worker 의존성 초기화 완료")


async def close_worker_dependencies() -> None:
    global _outfit_service

    logger.info("Worker 의존성 정리 시작...")

    _outfit_service = None
    await close_databases()

    logger.info("Worker 의존성 정리 완료")


def get_outfit_service_for_worker() -> OutfitService:
    if _outfit_service is None:
        raise RuntimeError("init_worker_dependencies()를 먼저 호출하세요")
    return _outfit_service
