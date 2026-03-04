"""
Kafka 클라이언트 모듈 (범용)
============================
AI 서버용 Kafka Producer / Consumer 초기화·종료·헬스체크.

- Producer: 결과 이벤트를 임의의 토픽에 발행 (공통, 싱글톤)
- Consumer: 팩토리 함수 create_consumer()로 토픽별 생성 (호출자가 토픽/그룹 지정)
"""

from __future__ import annotations

import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── 모듈 레벨 싱글톤 ──
_producer: AIOKafkaProducer | None = None
_consumers: list[AIOKafkaConsumer] = []


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 초기화 / 종료
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def init_kafka() -> None:
    """Kafka Producer를 초기화하고 브로커에 연결합니다."""
    global _producer
    settings = get_settings()

    logger.info(f"Connecting Kafka Producer to {settings.kafka_bootstrap_servers}")
    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=None,
    )
    await _producer.start()
    logger.info("Kafka Producer started")


async def create_consumer(topic: str, group_id: str) -> AIOKafkaConsumer:
    """범용 Consumer 팩토리 — 호출자가 토픽과 그룹을 지정하여 생성합니다.

    Args:
        topic: 구독할 Kafka 토픽 이름
        group_id: 컨슈머 그룹 ID

    Returns:
        시작된 AIOKafkaConsumer 인스턴스
    """
    settings = get_settings()

    logger.info(
        f"Connecting Kafka Consumer to {settings.kafka_bootstrap_servers} "
        f"(group={group_id}, topic={topic})"
    )
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=group_id,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
        max_poll_records=1,
    )
    await consumer.start()
    _consumers.append(consumer)
    logger.info(f"Kafka Consumer started (topic={topic})")
    return consumer


async def close_kafka() -> None:
    """Kafka Producer 및 등록된 모든 Consumer를 정상 종료합니다."""
    global _producer

    for consumer in _consumers:
        try:
            await consumer.stop()
            logger.info("Kafka Consumer stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka Consumer: {e}")
    _consumers.clear()

    if _producer:
        try:
            await _producer.stop()
            logger.info("Kafka Producer stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka Producer: {e}")
        finally:
            _producer = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 접근자
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_kafka_producer() -> AIOKafkaProducer:
    """초기화된 Kafka Producer를 반환합니다."""
    if _producer is None:
        raise RuntimeError(
            "Kafka Producer is not initialized. Call init_kafka() first."
        )
    return _producer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 헬스 체크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def check_kafka_health() -> str:
    """Kafka 연결 상태를 확인합니다."""
    try:
        if _producer is None:
            return "not_initialized"
        brokers = _producer.client.cluster.brokers()
        if not brokers:
            return "no_brokers"
        node_id = next(iter(brokers)).nodeId
        await _producer.client.ready(node_id)
        return "connected"
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        return f"error: {e}"
