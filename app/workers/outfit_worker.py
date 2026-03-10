import asyncio
import json
import logging

from aiokafka import ConsumerRecord

from app.common.kafka.schemas import (
    ErrorDetail,
    ErrorResponse,
    OutfitMetadata,
    OutfitRequestMessage,
    OutfitResponseMessage,
)
from app.common.kafka.serialization import deserialize, serialize
from app.workers.base import BaseWorker
from app.workers.config import worker_settings

logger = logging.getLogger(__name__)


class OutfitWorker(BaseWorker):
    def __init__(self) -> None:
        super().__init__(
            bootstrap_servers=worker_settings.KAFKA_BOOTSTRAP_SERVERS,
            group_id=worker_settings.OUTFIT_GROUP_ID,
            consume_topic=worker_settings.OUTFIT_REQUEST_TOPIC,
            produce_topic=worker_settings.OUTFIT_RESPONSE_TOPIC,
        )

    async def process_message(self, msg: ConsumerRecord) -> None:
        request = deserialize(msg.value, OutfitRequestMessage)

        await self.send_progress(request.request_id, "query_parsing", "의도 분석 중...")

        logger.info(f"파이프라인 실행 중... (User: {request.user_id})")
        await asyncio.sleep(1)

        response = OutfitResponseMessage(
            request_id=request.request_id,
            status="success",
            outfits=[],
            metadata=OutfitMetadata(confidence=0.9, processing_time_ms=1000),
        )
        await self.producer.send_and_wait(self.produce_topic, serialize(response))

    async def _handle_failure(self, msg: ConsumerRecord, error: Exception) -> None:  # noqa: ANN401
        request_id = "unknown"
        try:
            raw_data = json.loads(msg.value.decode("utf-8"))
            request_id = raw_data.get("requestId", "unknown")
        except Exception:
            logger.warning("메시지에서 requestId 추출 실패")

        error_res = ErrorResponse(
            request_id=request_id,
            status="failed",
            error=ErrorDetail(code="PROCESSING_ERROR", message=str(error)),
        )
        await self.producer.send_and_wait(self.produce_topic, serialize(error_res))
        # TODO: 최종 실패 시 outfit-dlq 토픽으로 전송 로직 추가 예정


if __name__ == "__main__":
    worker = OutfitWorker()
    asyncio.run(worker.run())
