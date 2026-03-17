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
from app.outfit.session_manager import SessionManager
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
        self.session_manager = SessionManager()  # Redis 기반 세션 관리자

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
            f"user_id={message.user_id}, query={message.query}, "
            f"session_id={message.session_id}"
        )

        # Progress 발행: 시작
        await self.send_progress(
            request_id=message.request_id,
            step=1,
            step_label="코디 추천 요청 처리 시작",
        )

        # 1. 세션 로드
        session_data = None
        if message.session_id:
            session_data = await self.session_manager.load_session(message.session_id)

        # 2. Progress 발행 콜백 정의
        async def send_progress_callback(step: int, step_label: str) -> None:
            await self.send_progress(
                request_id=message.request_id,
                step=step,
                step_label=step_label,
            )

        # 3. OutfitRequestMessage → OutfitRequest 변환
        outfit_request = OutfitRequest(
            user_id=message.user_id,
            query=message.query,
            session_id=message.session_id,
            urls=message.upload_slots,
            weather=None,
        )

        # 4. OutfitService 호출 (실제 LangGraph)
        service = get_outfit_service_for_worker()
        start_time = time.time()

        response = await service.recommend(
            request=outfit_request,
            trace_id=message.request_id,
            session_data=session_data,
            progress_callback=send_progress_callback,
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # 5. 세션 저장
        if message.session_id:
            # 대화 히스토리에 추가
            await self.session_manager.append_turn(
                session_id=message.session_id,
                user_id=message.user_id,
                user_message=message.query,
                assistant_message=response.query_summary,
            )

            # previous_outfits 업데이트
            session_data = await self.session_manager.load_session(message.session_id)
            if session_data:
                session_data.previous_outfits = response.outfits
                await self.session_manager.save_session(session_data)

        logger.info(
            f"코디 추천 완료: request_id={message.request_id}, "
            f"processing_time_ms={processing_time_ms}, "
            f"outfit_count={len(response.outfits)}"
        )

        # 6. OutfitResponse → OutfitResponseMessage 변환
        return OutfitResponseMessage(
            request_id=message.request_id,
            status="success",
            outfits=response.outfits,
            query_summary=response.query_summary,
            session_id=response.session_id,
            metadata=ResponseMetadata(
                processing_time_ms=processing_time_ms,
                model_version="v1",
                shop_supplemented=response.shop_supplemented
                if hasattr(response, "shop_supplemented")
                else False,
                fallback_used=response.fallback_used
                if hasattr(response, "fallback_used")
                else False,
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
