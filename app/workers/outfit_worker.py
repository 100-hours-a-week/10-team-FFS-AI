"""Outfit 추천 요청을 처리하는 Kafka Worker"""

from __future__ import annotations

import asyncio
import logging
import time

from app.common.kafka.schemas import (
    OutfitRequestMessage,
    OutfitResponseMessage,
    ResponseMetadata,
)
from app.outfit.schemas import OutfitRequest
from app.workers.base import BaseWorker
from app.workers.config import OutfitWorkerConfig, get_outfit_worker_config
from app.workers.dependencies import (
    close_worker_dependencies,
    get_outfit_service_for_worker,
    init_worker_dependencies,
)

logger = logging.getLogger(__name__)


class OutfitWorker(BaseWorker[OutfitRequestMessage]):
    """Outfit 추천 요청을 처리하는 Worker

    Kafka의 outfit-request 토픽에서 메시지를 받아
    OutfitService를 통해 코디 추천을 수행하고,
    결과를 outfit-response 토픽으로 발행합니다.
    """

    request_model_class = OutfitRequestMessage

    def __init__(self, config: OutfitWorkerConfig) -> None:
        super().__init__(config)
        self.worker_config = config

    @property
    def request_topic(self) -> str:
        """요청 토픽 이름"""
        return self.worker_config.request_topic

    @property
    def response_topic(self) -> str:
        """응답 토픽 이름"""
        return self.worker_config.response_topic

    @property
    def dlq_topic(self) -> str:
        """Dead Letter Queue 토픽 이름"""
        return self.worker_config.dlq_topic

    @property
    def group_id(self) -> str:
        """Consumer Group ID"""
        return self.worker_config.group_id

    async def process_message(
        self, message: OutfitRequestMessage
    ) -> OutfitResponseMessage:
        """Kafka 메시지를 받아서 OutfitService로 처리

        Args:
            message: Kafka에서 수신한 OutfitRequestMessage

        Returns:
            OutfitResponseMessage: 처리 결과
        """
        logger.info(
            f"코디 추천 시작: request_id={message.request_id}, "
            f"user_id={message.user_id}, query={message.query}"
        )

        # Progress 발행: 시작
        await self.send_progress(
            request_id=message.request_id,
            step=1,
            step_label="코디 추천 요청 처리 시작",
        )

        # OutfitRequestMessage → OutfitRequest 변환
        outfit_request = OutfitRequest(
            user_id=message.user_id,
            query=message.query,
            session_id=message.session_id,
            urls=message.upload_slots,
            weather=None,  # 현재 message에 weather 필드 없음
        )

        # OutfitService 호출
        service = get_outfit_service_for_worker()
        start_time = time.time()

        response = await service.recommend(
            request=outfit_request,
            trace_id=message.request_id,  # requestId를 trace_id로 사용 (Langfuse)
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        logger.info(
            f"코디 추천 완료: request_id={message.request_id}, "
            f"processing_time_ms={processing_time_ms}, "
            f"outfit_count={len(response.outfits)}"
        )

        # OutfitResponse → OutfitResponseMessage 변환
        return OutfitResponseMessage(
            request_id=message.request_id,
            status="completed",
            outfits=response.outfits,
            query_summary=response.query_summary,
            session_id=response.session_id,
            metadata=ResponseMetadata(
                processing_time_ms=processing_time_ms,
                model_version="v1",
            ),
        )


async def main() -> None:
    """Worker 진입점"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("OutfitWorker 초기화 시작...")

    # 의존성 초기화 (Qdrant, Redis, OutfitService)
    await init_worker_dependencies()

    # Worker 생성 및 시작
    config = get_outfit_worker_config()
    worker = OutfitWorker(config)

    try:
        await worker.start()
        logger.info("OutfitWorker 시작 완료, 메시지 수신 대기...")
        await worker.run()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt 수신, 종료 시작...")
    except Exception as e:
        logger.error(f"Worker 실행 중 오류: {e}", exc_info=True)
        raise
    finally:
        await worker.stop()
        await close_worker_dependencies()
        logger.info("OutfitWorker 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())
