from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiokafka import ConsumerRecord

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

# 테스트용 설정 상수
KAFKA_BOOTSTRAP_SERVERS = "localhost:9094"
TEST_REQUEST_TOPIC = "test-outfit-request"
TEST_RESPONSE_TOPIC = "test-outfit-response"
TEST_DLQ_TOPIC = "test-outfit-dlq"
TEST_GROUP_ID = "test-outfit-worker-group"


def make_consumer_record(
    value: bytes, topic: str = TEST_REQUEST_TOPIC
) -> ConsumerRecord:
    """테스트용 ConsumerRecord를 생성하는 헬퍼 함수.

    실제 Kafka 연결 없이 ConsumerRecord 객체를 직접 생성합니다.
    """
    return ConsumerRecord(
        topic=topic,
        partition=0,
        offset=0,
        timestamp=0,
        timestamp_type=0,
        key=None,
        value=value,
        checksum=None,
        serialized_key_size=-1,
        serialized_value_size=len(value),
        headers=(),
    )


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
async def mock_producer() -> AsyncMock:
    """Kafka Producer mock fixture.

    실제 Kafka 연결 없이 Producer 동작을 시뮬레이션합니다.
    """
    producer = AsyncMock()
    producer.start = AsyncMock()
    producer.stop = AsyncMock()
    producer.send_and_wait = AsyncMock()
    return producer


@pytest_asyncio.fixture
async def worker(
    worker_config: OutfitWorkerConfig, mock_producer: AsyncMock
) -> OutfitWorker:
    """OutfitWorker fixture.

    모든 외부 의존성(Kafka, Qdrant, Redis, SessionManager)을 mock하여
    인프라 없이 Worker 동작을 테스트합니다.
    """
    # SessionManager는 __init__에서 get_redis_client()를 호출하므로,
    # OutfitWorker 생성 후 session_manager를 AsyncMock으로 교체합니다.
    mock_session_manager = AsyncMock()
    mock_session_manager.load_session = AsyncMock(return_value=None)
    mock_session_manager.save_session = AsyncMock()
    mock_session_manager.append_turn = AsyncMock()

    # AIOKafkaConsumer는 start()/stop()이 async이므로 AsyncMock 필요
    mock_consumer = AsyncMock()
    mock_consumer.start = AsyncMock()
    mock_consumer.stop = AsyncMock()
    mock_consumer.commit = AsyncMock()

    with (
        # DB 초기화 mock (init_databases 안에서 Qdrant/Redis/Checkpointer 연결 방지)
        patch("app.workers.dependencies.init_databases", new_callable=AsyncMock),
        patch("app.workers.dependencies.close_databases", new_callable=AsyncMock),
        # SessionManager 생성 시 get_redis_client() 호출 방지
        patch("app.outfit.session_manager.get_redis_client", return_value=AsyncMock()),
        # dependencies의 get_redis_client와 TokenBucket Mock
        patch("app.workers.dependencies.get_redis_client", return_value=MagicMock()),
        patch("app.workers.dependencies.TokenBucket", return_value=MagicMock()),
        # BaseWorker.start()의 Kafka 연결 mock
        # return_value를 지정하여 AIOKafkaProducer()/AIOKafkaConsumer() 호출 시 mock 반환
        patch("app.workers.base.AIOKafkaProducer", return_value=mock_producer),
        patch("app.workers.base.AIOKafkaConsumer", return_value=mock_consumer),
    ):
        # Worker 의존성 초기화 (mock된 init_databases 호출)
        from app.workers.dependencies import init_worker_dependencies

        await init_worker_dependencies()

        outfit_worker = OutfitWorker(worker_config)

        # Redis 없이 세션 처리 가능하도록 session_manager 교체
        outfit_worker.session_manager = mock_session_manager

        # signal handler 등록은 pytest 루프에서 문제가 없으나,
        # 실제 루프의 add_signal_handler를 no-op mock으로 교체하여 안전하게 처리
        import asyncio as _asyncio

        loop = _asyncio.get_event_loop()
        original_add_signal_handler = loop.add_signal_handler
        loop.add_signal_handler = MagicMock()  # type: ignore[method-assign]

        try:
            await outfit_worker.start()

            # mock_producer를 _producer에 직접 주입 (send_and_wait 캡쳐용)
            outfit_worker._producer = mock_producer
            # mock_consumer를 _consumer에 직접 주입 (commit 캡쳐용)
            outfit_worker._consumer = mock_consumer

            yield outfit_worker

            await outfit_worker.stop()

            from app.workers.dependencies import close_worker_dependencies

            await close_worker_dependencies()
        finally:
            # 루프의 signal handler 복구
            loop.add_signal_handler = original_add_signal_handler  # type: ignore[method-assign]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_normal_flow_e2e(
    worker: OutfitWorker,
    mock_producer: AsyncMock,
) -> None:
    """정상 요청 → OutfitResponseMessage 발행 확인.

    실제 Kafka 없이 _process_record() 직접 호출 방식으로 검증합니다.
    """
    mock_service = AsyncMock()
    mock_service.recommend = AsyncMock(
        return_value=MagicMock(
            outfits=[],
            query_summary="테스트 코디 추천",
            session_id="test-session",
            shop_supplemented=False,
            fallback_used=False,
        )
    )

    request_message = OutfitRequestMessage(
        request_id="test-e2e-123",
        user_id=1,
        query="캐주얼 코디 추천해줘",
        session_id="test-session",
    )
    record = make_consumer_record(serialize(request_message))

    with patch(
        "app.workers.outfit_worker.get_outfit_service_for_worker",
        return_value=mock_service,
    ):
        await worker._process_record(record)

    # send_and_wait 호출 내역에서 OutfitResponseMessage 찾기
    # 환경변수 override 가능성: 상수 대신 worker.response_topic 사용
    response_call = None
    for call in mock_producer.send_and_wait.call_args_list:
        args = call[0]
        if args[0] == worker.response_topic:
            try:
                candidate = deserialize(args[1], OutfitResponseMessage)
                if (
                    candidate.request_id == "test-e2e-123"
                    and candidate.status == "success"
                ):
                    response_call = candidate
                    break
            except Exception:
                continue

    assert response_call is not None, "OutfitResponseMessage를 발행하지 않았습니다"
    assert response_call.request_id == "test-e2e-123"
    assert response_call.status == "success"
    assert response_call.query_summary == "테스트 코디 추천"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invalid_json_to_dlq(
    worker: OutfitWorker,
    mock_producer: AsyncMock,
) -> None:
    """잘못된 JSON 메시지 → DLQ 발행 확인.

    역직렬화 실패 시 DLQ에 DeserializationError 메시지가 발행되어야 합니다.
    """
    invalid_json = b'{"invalid json without closing brace'
    record = make_consumer_record(invalid_json)

    await worker._process_record(record)

    # send_and_wait 호출 내역에서 DLQMessage 찾기
    # 환경변수 override 가능성: 상수 대신 worker.dlq_topic 사용
    dlq_call = None
    for call in mock_producer.send_and_wait.call_args_list:
        args = call[0]
        if args[0] == worker.dlq_topic:
            dlq_call = deserialize(args[1], DLQMessage)
            break

    assert dlq_call is not None, "DLQ 메시지를 발행하지 않았습니다"
    assert dlq_call.original_topic == worker.request_topic
    assert dlq_call.error.type == "DeserializationError"
    assert dlq_call.retry_count == 0  # 영구 실패 → 재시도 없이 즉시 DLQ
    assert "JSON" in dlq_call.error.message or "json" in dlq_call.error.message


@pytest.mark.integration
@pytest.mark.asyncio
async def test_process_failure_to_dlq_and_error_response(
    worker: OutfitWorker,
    mock_producer: AsyncMock,
) -> None:
    """process_message 강제 실패 → DLQ + ErrorResponse 확인.

    OutfitService가 InfrastructureError를 발생시키도록 mock →
    재시도 2회 후 DLQ + ErrorResponse 발행 확인
    """
    mock_service = AsyncMock()
    mock_service.recommend = AsyncMock(
        side_effect=InfrastructureError("Qdrant timeout", service="qdrant")
    )

    request_message = OutfitRequestMessage(
        request_id="test-failure-456",
        user_id=2,
        query="실패 테스트",
        session_id="test-session-2",
    )
    record = make_consumer_record(serialize(request_message))

    with patch(
        "app.workers.outfit_worker.get_outfit_service_for_worker",
        return_value=mock_service,
    ):
        await worker._process_record(record)

    # recommend가 3번 호출되어야 함 (첫 시도 + 재시도 2회)
    assert mock_service.recommend.call_count == 3

    # DLQ 메시지 검증
    # 환경변수 override 가능성: 상수 대신 worker.dlq_topic 사용
    dlq_call = None
    for call in mock_producer.send_and_wait.call_args_list:
        args = call[0]
        if args[0] == worker.dlq_topic:
            dlq_call = deserialize(args[1], DLQMessage)
            break

    assert dlq_call is not None, "DLQ 메시지를 발행하지 않았습니다"
    assert dlq_call.original_topic == worker.request_topic
    assert dlq_call.error.type == "InfrastructureError"
    assert dlq_call.retry_count == 2

    # ErrorResponse 메시지 검증
    # 환경변수 override 가능성: 상수 대신 worker.response_topic 사용
    error_response_call = None
    for call in mock_producer.send_and_wait.call_args_list:
        args = call[0]
        if args[0] == worker.response_topic:
            try:
                candidate = deserialize(args[1], ErrorResponse)
                if candidate.request_id == "test-failure-456":
                    error_response_call = candidate
                    break
            except Exception:
                continue

    assert error_response_call is not None, "ErrorResponse를 발행하지 않았습니다"
    assert error_response_call.request_id == "test-failure-456"
    assert error_response_call.status == "failed"
    assert error_response_call.error.code == "INFRASTRUCTURE_ERROR"
    assert error_response_call.error.retry_after_seconds == 30
