"""
Kafka 클라이언트 모듈
====================
AI 서버용 Kafka Producer(결과 발행) / Consumer(요청 수신) 초기화·종료·헬스체크.

- Producer: 분석 결과 이벤트를 `ai.clothes.analyze.result` 토픽에 발행
- Consumer: `ai.clothes.analyze.request` 토픽을 구독하여 분석 요청 수신
"""

from __future__ import annotations

import logging

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── 모듈 레벨 싱글톤 ──
_producer: AIOKafkaProducer | None = None
_consumer: AIOKafkaConsumer | None = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 초기화 / 종료
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def init_kafka(is_worker: bool = False) -> None:
    """Kafka Producer와 Consumer를 초기화하고 브로커에 연결합니다."""
    global _producer, _consumer
    settings = get_settings()

    # 1. Producer (결과 토픽에 이벤트 발행용)
    logger.info(f"Connecting Kafka Producer to {settings.kafka_bootstrap_servers}")
    _producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        # JSON bytes를 직접 넘기므로 별도 serializer 불필요
        value_serializer=None,
    )
    await _producer.start()
    logger.info("Kafka Producer started")

    # 2. Consumer (요청 토픽 구독용 - 워커에서만 실행)
    if is_worker:
        logger.info(
            f"Connecting Kafka Consumer to {settings.kafka_bootstrap_servers} "
            f"(group={settings.kafka_consumer_group}, "
            f"topic={settings.kafka_closet_request_topic})"
        )
        _consumer = AIOKafkaConsumer(
            settings.kafka_closet_request_topic,
            bootstrap_servers=settings.kafka_bootstrap_servers,
            group_id=settings.kafka_consumer_group,
            auto_offset_reset="earliest",
            enable_auto_commit=False,  # 처리 완료 후 수동 커밋
            max_poll_records=1,  # 한 번에 1개씩 (메모리 보호)
        )
        await _consumer.start()
        logger.info("Kafka Consumer started")


async def close_kafka() -> None:
    """Kafka Producer/Consumer 연결을 정상 종료합니다."""
    global _producer, _consumer

    if _consumer:
        try:
            await _consumer.stop()
            logger.info("Kafka Consumer stopped")
        except Exception as e:
            logger.error(f"Error stopping Kafka Consumer: {e}")
        finally:
            _consumer = None

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


def get_kafka_consumer() -> AIOKafkaConsumer:
    """초기화된 Kafka Consumer를 반환합니다."""
    if _consumer is None:
        raise RuntimeError(
            "Kafka Consumer is not initialized. Call init_kafka() first."
        )
    return _consumer


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 헬스 체크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


async def check_kafka_health() -> str:
    """Kafka 연결 상태를 확인합니다."""
    try:
        if _producer is None:
            return "not_initialized"
        # Producer의 내부 클라이언트로 간단한 메타데이터 요청
        await _producer.client.ready(0)
        return "connected"
    except Exception as e:
        logger.error(f"Kafka health check failed: {e}")
        return f"error: {e}"
