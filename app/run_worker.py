"""KlosetLab - Dedicated Kafka Consumer Worker

이 스크립트는 FastAPI 서버와 독립적으로 실행되며,
오직 Kafka 메시지를 수신(Consume)하고 비즈니스 로직(AI 분석 등)을 처리하는 역할만 수행합니다.

실행 방법:
    python -m app.run_worker
"""

import asyncio
import logging
import signal

from app.closet.handler import create_handler
from app.config import get_settings
from app.core.consumer import consume_loop
from app.core.database import close_databases, init_databases
from app.core.kafka import close_kafka, create_consumer, init_kafka

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("worker")


async def main() -> None:
    logger.info("Worker startup starting...")

    # 1. 의존성 초기화 (DB, Kafka Producer)
    await init_databases()
    await init_kafka()

    # 2. Closet 분석용 Consumer 생성
    settings = get_settings()
    closet_consumer = await create_consumer(
        topic=settings.kafka_closet_request_topic,
        group_id=settings.kafka_consumer_group,
    )
    closet_handler = create_handler(closet_consumer)

    # ─── 나중에 다른 기능 추가 시 아래처럼 확장 ───
    # embedding_consumer = await create_consumer(
    #     topic=settings.kafka_embedding_request_topic,
    #     group_id="ai_embedding_worker_group",
    # )
    # embedding_handler = create_handler_embedding(embedding_consumer)
    # asyncio.create_task(consume_loop(embedding_consumer, embedding_handler))

    logger.info("Worker startup complete. Listening for messages...")

    # 3. 메인 루프 실행
    consumer_task = asyncio.create_task(consume_loop(closet_consumer, closet_handler))

    # 4. 우아한 종료(Graceful Shutdown) 처리를 위한 이벤트
    stop_event = asyncio.Event()

    def handle_sigterm() -> None:
        logger.info("Received termination signal. Shutting down worker...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, handle_sigterm)

    # 종료 신호를 받을 때까지 대기
    await stop_event.wait()

    # 5. 루프 취소 및 리소스 정리
    consumer_task.cancel()
    try:
        await consumer_task
    except asyncio.CancelledError:
        pass

    await close_kafka()
    await close_databases()
    logger.info("Worker shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
