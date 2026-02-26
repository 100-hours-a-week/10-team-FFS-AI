"""Closet 분석 요청 핸들러 — Kafka 메시지 처리 오케스트레이터

이 모듈은 Kafka 통신만 전담합니다:
- 메시지 수신(ConsumerRecord) → 역직렬화
- ClosetService에 비즈니스 로직 위임
- 결과 이벤트를 Kafka 결과 토픽에 발행
- 오프셋 커밋
"""

from __future__ import annotations

import logging

from aiokafka import ConsumerRecord

from app.closet.events import (
    AnalyzedPayload,
    AnalyzingCompletedEvent,
    PreprocessingCompletedEvent,
    PreprocessPayload,
    deserialize_request_event,
    serialize_event,
)
from app.closet.service import ClosetService
from app.config import get_settings
from app.core.kafka import get_kafka_consumer, get_kafka_producer

logger = logging.getLogger(__name__)

# 비즈니스 로직 전담 서비스 (싱글톤)
_service = ClosetService()


async def handle_analysis_request(message: ConsumerRecord) -> None:
    """Kafka 메시지 1건을 수신하여 ClosetService에 처리를 위임하고 결과를 발행합니다."""
    settings = get_settings()
    producer = get_kafka_producer()
    consumer = get_kafka_consumer()

    # 1. 메시지 역직렬화
    event = deserialize_request_event(message.value)
    req = event.data
    logger.info(f"요청 수신: batch={req.batch_id}, task={req.task_id}")

    try:
        # ── 전처리 단계: 이미지 다운로드 + S3 업로드 ──
        preprocess_result = await _service.preprocess(
            target_image_url=req.target_image,
            user_id=req.user_id,
        )

        if not preprocess_result.success:
            logger.error(
                f"전처리 실패, 건너뜀: task={req.task_id}, "
                f"error={preprocess_result.error}"
            )
            await consumer.commit()
            return

        # PREPROCESSING_COMPLETED 이벤트 발행
        preprocess_event = PreprocessingCompletedEvent(
            requested_at=event.requested_at,
            data=PreprocessPayload(
                batchId=req.batch_id,
                taskId=req.task_id,
                fileId=preprocess_result.file_id,
            ),
        )
        await producer.send_and_wait(
            settings.kafka_result_topic,
            serialize_event(preprocess_event),
        )
        logger.info(f"전처리 완료: task={req.task_id}")

        # ── 분석 단계: VLM 이미지 분석 ──
        analysis_result = await _service.analyze(
            target_image_url=req.target_image,
        )

        # ANALYZING_COMPLETED 이벤트 발행
        analyze_event = AnalyzingCompletedEvent(
            requested_at=event.requested_at,
            data=AnalyzedPayload(
                batchId=req.batch_id,
                taskId=req.task_id,
                major=analysis_result.major,
                extra=analysis_result.extra,
            ),
        )
        await producer.send_and_wait(
            settings.kafka_result_topic,
            serialize_event(analyze_event),
        )
        logger.info(f"분석 완료: task={req.task_id}")

        # 오프셋 커밋
        await consumer.commit()

    except Exception as e:
        logger.error(f"처리 실패: batch={req.batch_id}, task={req.task_id}, error={e}")
        await consumer.commit()
