import logging

from app.core.database import close_databases, init_databases
from app.outfit.service import OutfitService

logger = logging.getLogger(__name__)

_outfit_service: OutfitService | None = None


async def init_worker_dependencies() -> None:
    global _outfit_service

    logger.info("Worker 의존성 초기화 시작...")

    await init_databases()

    _outfit_service = OutfitService()

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
