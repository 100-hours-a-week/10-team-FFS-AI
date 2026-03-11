import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.closet.handler import create_handler as create_closet_handler
from app.config import (
    ConsumerConfig,
    KafkaTopics,
    get_settings,
)
from app.core.consumer import consume_loop
from app.core.database import check_health, close_databases, init_databases
from app.core.kafka import (
    check_kafka_health,
    close_kafka,
    create_consumer,
    init_kafka,
)
from app.embedding.router import router as embedding_router
from app.outfit.router import router as outfit_router
from app.shop.router import router as shop_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

settings = get_settings()

# ── 토픽별 컨슈머 설정 ──
# 새 기능 추가 시 이 리스트에 ConsumerConfig를 추가하면 됩니다.
CONSUMER_CONFIGS: list[ConsumerConfig] = [
    ConsumerConfig(
        topic=KafkaTopics.CLOTHES_ANALYZE_REQUEST,
        group_id=settings.kafka_group_id_clothes_analyze,
        handler_factory=create_closet_handler,
        replicas=3,
    ),
]

# 실행 중인 컨슈머 태스크를 추적 (종료 시 정리용)
_consumer_tasks: list[asyncio.Task] = []


def _init_langfuse() -> None:
    settings = get_settings()
    if not settings.langfuse_enabled:
        logger.info("Langfuse disabled (LANGFUSE_ENABLED=false)")
        return

    if not settings.langfuse_secret_key or not settings.langfuse_public_key:
        logger.warning("Langfuse enabled but keys not set, skipping initialization")
        return

    from langfuse import Langfuse

    Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_base_url,
    )
    logger.info("Langfuse initialized (host=%s)", settings.langfuse_base_url)


def _shutdown_langfuse() -> None:
    settings = get_settings()
    if not settings.langfuse_enabled:
        return

    try:
        from langfuse import get_client

        langfuse = get_client()
        langfuse.flush()
        logger.info("Langfuse flushed successfully")
    except Exception as e:
        logger.warning("Langfuse flush failed: %s", e)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("start server")
    try:
        await init_databases()
        _init_langfuse()
        logger.info("Database connections initialized")
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        raise

    # Kafka 연결 실패 시에도 HTTP API는 정상 동작
    kafka_ready = False
    try:
        await init_kafka()
        logger.info("Kafka Producer initialized")
        kafka_ready = True
    except Exception as e:
        logger.warning(f"Kafka 연결 실패 (HTTP API는 정상 동작): {e}")

    # Kafka가 준비된 경우에만 컨슈머 시작
    if kafka_ready:
        for config in CONSUMER_CONFIGS:
            for i in range(config.replicas):
                consumer = await create_consumer(config.topic, config.group_id)
                handler = config.handler_factory(consumer)
                task = asyncio.create_task(
                    consume_loop(consumer, handler),
                    name=f"consumer-{config.topic}-{i}",
                )
                _consumer_tasks.append(task)
                logger.info(
                    f"Consumer 시작: topic={config.topic}, "
                    f"replica={i + 1}/{config.replicas}"
                )

    logger.info("Application startup complete")

    yield

    logger.info("shut down server")
    try:
        # 컨슈머 태스크 정리
        for task in _consumer_tasks:
            task.cancel()
        if _consumer_tasks:
            await asyncio.gather(*_consumer_tasks, return_exceptions=True)
            logger.info(f"Consumer {len(_consumer_tasks)}개 종료 완료")
        _consumer_tasks.clear()

        await close_kafka()
        _shutdown_langfuse()
        await close_databases()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="KlosetLab",
    lifespan=lifespan,
)


Instrumentator().instrument(app).expose(app)


app.include_router(embedding_router, prefix="/ai")
app.include_router(outfit_router, prefix="/ai")
app.include_router(shop_router, prefix="/ai")


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/health")
async def health_check() -> JSONResponse:
    health_status = await check_health()
    health_status["kafka"] = await check_kafka_health()

    all_connected = all(status == "connected" for status in health_status.values())
    overall_status = "healthy" if all_connected else "degraded"

    response_data = {
        "status": overall_status,
        "services": health_status,
    }
    status_code = 200 if all_connected else 503

    return JSONResponse(content=response_data, status_code=status_code)
