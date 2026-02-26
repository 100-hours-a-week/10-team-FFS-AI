"""공용 Kafka Consumer 루프 — 메시지 수신 및 핸들러 분배"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine

from aiokafka import AIOKafkaConsumer, ConsumerRecord

from app.config import get_settings

logger = logging.getLogger(__name__)

# 핸들러 타입: Kafka 메시지를 받아서 처리하는 비동기 함수
MessageHandler = Callable[..., Coroutine[object, object, None]]


async def consume_loop(consumer: AIOKafkaConsumer, handler: MessageHandler) -> None:
    """Kafka 토픽을 무한히 구독하며 메시지를 핸들러에 전달하는 공용 루프.

    Args:
        consumer: 시작된 AIOKafkaConsumer 인스턴스
        handler: 메시지 1건을 처리하는 비동기 함수 (예: closet의 handle_analysis_request)
    """
    settings = get_settings()
    semaphore = asyncio.Semaphore(settings.kafka_max_concurrent_tasks)

    logger.info(
        f"Consumer 루프 시작 (max_concurrent_tasks={settings.kafka_max_concurrent_tasks})"
    )

    try:
        async for message in consumer:
            await semaphore.acquire()
            asyncio.create_task(_process_and_release(message, handler, semaphore))
    except asyncio.CancelledError:
        logger.info("Consumer 루프 정상 종료")
    except Exception as e:
        logger.error(f"Consumer 루프 예외: {e}")
        raise


async def _process_and_release(
    message: ConsumerRecord, handler: MessageHandler, semaphore: asyncio.Semaphore
) -> None:
    """메시지 처리 후 세마포어를 반드시 반환하는 래퍼."""
    try:
        await handler(message)
    except Exception as e:
        logger.error(f"메시지 처리 실패: {e}")
    finally:
        semaphore.release()
