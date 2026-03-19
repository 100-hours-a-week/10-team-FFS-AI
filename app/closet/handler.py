"""Closet 분석 요청 핸들러 — Kafka 메시지 처리 오케스트레이터

이 모듈은 Kafka 통신만 전담합니다:
- 메시지 수신(ConsumerRecord) → 역직렬화
- ClosetService에 비즈니스 로직 위임
- 결과 이벤트를 Kafka 결과 토픽에 발행
- 오프셋 커밋

이벤트 발행 순서:
1. ABUSING_COMPLETED (항상 passed=true)
2. SEGMENTATION_COMPLETED (totalItems)
3. PREPROCESSING_COMPLETED × N (아이템별)
4. ANALYZING_COMPLETED × N (아이템별)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine

from aiokafka import AIOKafkaConsumer, ConsumerRecord

from app.closet.events import (
    AbusingCompletedEvent,
    AbusingPayload,
    AnalyzedPayload,
    AnalyzingCompletedEvent,
    PreprocessingCompletedEvent,
    PreprocessPayload,
    SegmentationCompletedEvent,
    SegmentationPayload,
    deserialize_request_event,
    serialize_event,
)
from app.closet.service import ClosetService
from app.config import KafkaTopics
from app.core.kafka import get_kafka_producer

logger = logging.getLogger(__name__)

# 핸들러 타입
MessageHandler = Callable[..., Coroutine[object, object, None]]

# 비즈니스 로직 전담 서비스 (지연 초기화 — import 시점에 API 키 불필요)
_service: ClosetService | None = None


def _get_service() -> ClosetService:
    global _service
    if _service is None:
        _service = ClosetService()
    return _service


def create_handler(consumer: AIOKafkaConsumer) -> MessageHandler:
    """consumer를 바인딩한 핸들러 함수를 생성합니다.

    Args:
        consumer: 오프셋 커밋에 사용할 AIOKafkaConsumer 인스턴스

    Returns:
        메시지 1건을 처리하는 비동기 함수
    """

    async def handle_analysis_request(message: ConsumerRecord) -> None:
        """Kafka 메시지 1건을 수신하여 ClosetService에 처리를 위임하고 결과를 발행합니다."""
        producer = get_kafka_producer()
        topic = KafkaTopics.CLOTHES_ANALYZE_RESULT

        # 1. 메시지 역직렬화
        event = deserialize_request_event(message.value)
        req = event.data
        logger.info(f"요청 수신: batch={req.batch_id}, source={req.source_id}")

        try:
            # ── ① 어뷰징 검사 (Ray Serve 호출) ──
            validation_result = await _get_service().validate_image(req.target_image)

            passed = True
            error_reason = None

            if validation_result.get("error"):
                passed = False
                error_reason = f"VALIDATION_ERROR: {validation_result['error']}"
            elif validation_result.get("nsfw") and validation_result["nsfw"]["is_nsfw"]:
                passed = False
                error_reason = "NSFW"
            elif (
                validation_result.get("fashion")
                and not validation_result["fashion"]["is_fashion"]
            ):
                passed = False
                error_reason = "NOT_FASHION"

            abusing_event = AbusingCompletedEvent(
                requested_at=event.requested_at,
                data=AbusingPayload(
                    batchId=req.batch_id,
                    sourceId=req.source_id,
                    passed=passed,
                ),
            )
            key = req.source_id.encode("utf-8")
            await producer.send_and_wait(topic, serialize_event(abusing_event), key=key)

            if not passed:
                logger.warning(
                    f"어뷰징 검사 실패: source={req.source_id}, reason={error_reason}"
                )
                await consumer.commit()
                return

            logger.info(f"어뷰징 검사 통과: source={req.source_id}")

            # ── ② 세그멘테이션 + 전처리 ──
            preprocess_result = await _get_service().preprocess(
                target_image_url=req.target_image,
                user_id=req.user_id,
            )

            if not preprocess_result.success:
                logger.error(
                    f"전처리 실패, 건너뜀: source={req.source_id}, "
                    f"error={preprocess_result.error}"
                )
                await consumer.commit()
                return

            item_count = len(preprocess_result.file_ids)

            # SEGMENTATION_COMPLETED 발행
            seg_event = SegmentationCompletedEvent(
                requested_at=event.requested_at,
                data=SegmentationPayload(
                    batchId=req.batch_id,
                    sourceId=req.source_id,
                    segmentation=item_count,
                ),
            )
            await producer.send_and_wait(topic, serialize_event(seg_event), key=key)
            logger.info(
                f"세그멘테이션 완료: source={req.source_id}, items={item_count}"
            )

            # ── ③ 아이템별 PREPROCESSING_COMPLETED 발행 ──
            for i in range(item_count):
                pre_event = PreprocessingCompletedEvent(
                    requested_at=event.requested_at,
                    data=PreprocessPayload(
                        batchId=req.batch_id,
                        sourceId=req.source_id,
                        taskId=preprocess_result.task_ids[i],
                        fileId=preprocess_result.file_ids[i],
                    ),
                )
                await producer.send_and_wait(topic, serialize_event(pre_event), key=key)
                logger.info(
                    f"전처리 완료: task={preprocess_result.task_ids[i]}, "
                    f"file={preprocess_result.file_ids[i]}"
                )

            # ── ④ 아이템별 분석 (Gemma-3 VLM) - 병렬 처리 ──
            analysis_tasks = [
                _get_service().analyze(preprocess_result.item_images[i])
                for i in range(item_count)
            ]
            analysis_results = await asyncio.gather(*analysis_tasks)

            # ── ⑤ 분석 결과 이벤트 발행 (순서 보장을 위해 순차 발행) ──
            for i, analysis_result in enumerate(analysis_results):
                analyze_event = AnalyzingCompletedEvent(
                    requested_at=event.requested_at,
                    data=AnalyzedPayload(
                        batchId=req.batch_id,
                        sourceId=req.source_id,
                        taskId=preprocess_result.task_ids[i],
                        major=analysis_result.major,
                        extra=analysis_result.extra,
                    ),
                )
                await producer.send_and_wait(
                    topic, serialize_event(analyze_event), key=key
                )
                logger.info(
                    f"분석 완료: task={preprocess_result.task_ids[i]}, "
                    f"category={analysis_result.major.category}"
                )

            # 오프셋 커밋
            await consumer.commit()
            logger.info(
                f"처리 완료: batch={req.batch_id}, source={req.source_id}, "
                f"items={item_count}"
            )

        except Exception as e:
            logger.error(
                f"처리 실패: batch={req.batch_id}, source={req.source_id}, error={e}"
            )
            await consumer.commit()

    return handle_analysis_request
