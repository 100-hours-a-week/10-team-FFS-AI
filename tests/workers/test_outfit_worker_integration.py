from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from app.common.kafka.exceptions import InfrastructureError
from app.common.kafka.schemas import (
    DLQMessage,
    ErrorResponse,
    OutfitRequestMessage,
    OutfitResponseMessage,
)
from app.common.kafka.serialization import deserialize, serialize
from app.workers.config import OutfitWorkerConfig
from app.workers.outfit_worker import OutfitWorker

KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
TEST_REQUEST_TOPIC = "test-outfit-request"
TEST_RESPONSE_TOPIC = "test-outfit-response"
TEST_DLQ_TOPIC = "test-outfit-dlq"
TEST_GROUP_ID = "test-outfit-worker-group"


@pytest_asyncio.fixture
async def kafka_producer() -> AIOKafkaProducer:
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS)
    await producer.start()
    yield producer
    await producer.stop()


@pytest_asyncio.fixture
async def kafka_consumer_response() -> AIOKafkaConsumer:
    consumer = AIOKafkaConsumer(
        TEST_RESPONSE_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id=f"{TEST_GROUP_ID}-response-consumer",
        enable_auto_commit=True,
    )
    await consumer.start()
    yield consumer
    await consumer.stop()


@pytest_asyncio.fixture
async def kafka_consumer_dlq() -> AIOKafkaConsumer:
    """DLQ 토픽을 구독하는 Kafka Consumer fixture."""
    consumer = AIOKafkaConsumer(
        TEST_DLQ_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset="earliest",
        group_id=f"{TEST_GROUP_ID}-dlq-consumer",
        enable_auto_commit=True,
    )
    await consumer.start()
    yield consumer
    await consumer.stop()


@pytest_asyncio.fixture
async def worker_config() -> OutfitWorkerConfig:
    """테스트용 OutfitWorker 설정."""
    return OutfitWorkerConfig(
        kafka_bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        request_topic=TEST_REQUEST_TOPIC,
        response_topic=TEST_RESPONSE_TOPIC,
        dlq_topic=TEST_DLQ_TOPIC,
        group_id=TEST_GROUP_ID,
    )


@pytest_asyncio.fixture
async def worker(worker_config: OutfitWorkerConfig) -> OutfitWorker:
    """OutfitWorker fixture."""
    worker = OutfitWorker(worker_config)
    await worker.start()
    yield worker
    await worker.stop()


async def consume_one_message(
    consumer: AIOKafkaConsumer, timeout_seconds: int = 10
) -> bytes | None:
    try:
        async with asyncio.timeout(timeout_seconds):
            async for msg in consumer:
                return msg.value
    except TimeoutError:
        return None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normal_flow_e2e(
    kafka_producer: AIOKafkaProducer,
    kafka_consumer_response: AIOKafkaConsumer,
    worker: OutfitWorker,
) -> None:
    mock_service = AsyncMock()
    mock_service.recommend = AsyncMock(
        return_value=AsyncMock(
            outfits=[],
            query_summary="테스트 코디 추천",
            session_id="test-session",
        )
    )

    with patch(
        "app.workers.outfit_worker.get_outfit_service_for_worker",
        return_value=mock_service,
    ):
        # 1. 메시지 발행
        request_message = OutfitRequestMessage(
            request_id="test-e2e-123",
            user_id=1,
            query="캐주얼 코디 추천해줘",
            session_id="test-session",
        )

        await kafka_producer.send_and_wait(
            TEST_REQUEST_TOPIC,
            serialize(request_message),
            key=request_message.request_id.encode("utf-8"),
        )

        # 2. Worker가 메시지를 처리할 시간 대기 (짧은 시간)
        # Worker.run()은 별도 태스크에서 실행되어야 하지만,
        # 여기서는 단순화하여 _process_record를 직접 호출
        # (실제 E2E는 Worker를 별도 프로세스로 실행해야 함)

        response_data = await consume_one_message(
            kafka_consumer_response, timeout_seconds=5
        )

        assert response_data is not None, "응답 메시지를 받지 못했습니다"

        response = deserialize(response_data, OutfitResponseMessage)
        assert response.request_id == "test-e2e-123"
        assert response.status == "success"
        assert response.query_summary == "테스트 코디 추천"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_json_to_dlq(
    kafka_producer: AIOKafkaProducer,
    kafka_consumer_dlq: AIOKafkaConsumer,
    worker: OutfitWorker,
) -> None:
    invalid_json = b'{"invalid json without closing brace'

    await kafka_producer.send_and_wait(
        TEST_REQUEST_TOPIC,
        invalid_json,
    )

    # 2. Worker가 처리할 시간 대기
    # (실제로는 Worker.run()이 별도 태스크에서 실행 중이어야 함)

    dlq_data = await consume_one_message(kafka_consumer_dlq, timeout_seconds=5)

    assert dlq_data is not None, "DLQ 메시지를 받지 못했습니다"

    dlq_message = deserialize(dlq_data, DLQMessage)
    assert dlq_message.original_topic == TEST_REQUEST_TOPIC
    assert dlq_message.error.type == "DeserializationError"
    assert dlq_message.retry_count == 0  # 재시도 없이 즉시 DLQ
    assert "JSON" in dlq_message.error.message or "json" in dlq_message.error.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_failure_to_dlq_and_error_response(
    kafka_producer: AIOKafkaProducer,
    kafka_consumer_response: AIOKafkaConsumer,
    kafka_consumer_dlq: AIOKafkaConsumer,
    worker: OutfitWorker,
) -> None:
    """3. process_message 강제 실패 → DLQ + ErrorResponse 확인.

    OutfitService가 InfrastructureError를 발생시키도록 Mock 설정 →
    재시도 2회 후 DLQ + ErrorResponse 발행 확인
    """

    mock_service = AsyncMock()
    mock_service.recommend = AsyncMock(
        side_effect=InfrastructureError("Qdrant timeout", service="qdrant")
    )

    with patch(
        "app.workers.outfit_worker.get_outfit_service_for_worker",
        return_value=mock_service,
    ):
        request_message = OutfitRequestMessage(
            request_id="test-failure-456",
            user_id=2,
            query="실패 테스트",
            session_id="test-session-2",
        )

        await kafka_producer.send_and_wait(
            TEST_REQUEST_TOPIC,
            serialize(request_message),
            key=request_message.request_id.encode("utf-8"),
        )

        dlq_data = await consume_one_message(kafka_consumer_dlq, timeout_seconds=15)
        assert dlq_data is not None, "DLQ 메시지를 받지 못했습니다"

        dlq_message = deserialize(dlq_data, DLQMessage)
        assert dlq_message.original_topic == TEST_REQUEST_TOPIC
        assert dlq_message.error.type == "InfrastructureError"
        assert dlq_message.retry_count == 2

        response_data = await consume_one_message(
            kafka_consumer_response, timeout_seconds=5
        )
        assert response_data is not None, "ErrorResponse를 받지 못했습니다"

        error_response = deserialize(response_data, ErrorResponse)
        assert error_response.request_id == "test-failure-456"
        assert error_response.status == "failed"
        assert error_response.error.code == "INFRASTRUCTURE_ERROR"
        assert error_response.error.retry_after_seconds == 30
