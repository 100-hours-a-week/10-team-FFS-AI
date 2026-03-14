"""BaseWorker 재시도 로직 단위 테스트."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiokafka import ConsumerRecord

from app.common.kafka.exceptions import InfrastructureError, RateLimitError
from app.common.kafka.schemas import OutfitRequestMessage, OutfitResponseMessage
from app.common.schemas import BaseSchema
from app.workers.base import BaseWorker
from app.workers.config import WorkerConfig


class MockWorker(BaseWorker[OutfitRequestMessage]):
    """테스트용 Worker 구현 (Mock)."""

    request_model_class = OutfitRequestMessage

    def __init__(self, config: WorkerConfig) -> None:
        super().__init__(config)
        self._request_topic = "test-request"
        self._response_topic = "test-response"
        self._dlq_topic = "test-dlq"
        self._group_id = "test-group"

    @property
    def request_topic(self) -> str:
        return self._request_topic

    @property
    def response_topic(self) -> str:
        return self._response_topic

    @property
    def dlq_topic(self) -> str:
        return self._dlq_topic

    @property
    def group_id(self) -> str:
        return self._group_id

    async def process_message(self, message: OutfitRequestMessage) -> BaseSchema:
        """Mock으로 대체될 메서드."""
        return OutfitResponseMessage(
            request_id=message.request_id,
            status="completed",
            outfits=[],
        )


@pytest.fixture
def worker_config() -> WorkerConfig:
    """Worker 설정 fixture."""
    return WorkerConfig()


@pytest.fixture
def worker(worker_config: WorkerConfig) -> MockWorker:
    """MockWorker fixture."""
    worker = MockWorker(worker_config)
    # Mock Consumer/Producer 설정
    worker._consumer = AsyncMock()
    worker._producer = AsyncMock()
    return worker


@pytest.fixture
def mock_record() -> ConsumerRecord:
    """Mock Kafka ConsumerRecord fixture."""
    message = OutfitRequestMessage(
        request_id="test-123",
        user_id=1,
        query="코디 추천",
        session_id="session-1",
    )
    return ConsumerRecord(
        topic="test-request",
        partition=0,
        offset=100,
        timestamp=0,
        timestamp_type=0,
        key=None,
        value=message.model_dump_json(by_alias=True).encode("utf-8"),
        checksum=None,
        serialized_key_size=0,
        serialized_value_size=0,
        headers=[],
    )


class TestBaseWorkerRetryLogic:
    """BaseWorker 재시도 로직 테스트."""

    @pytest.mark.asyncio
    async def test_successful_processing_no_retry(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """1. 정상 처리 (재시도 없음)."""
        # process_message Mock 설정 (성공)
        worker.process_message = AsyncMock(
            return_value=OutfitResponseMessage(
                request_id="test-123",
                status="completed",
                outfits=[],
            )
        )

        # 실행
        await worker._process_record(mock_record)

        # 검증
        assert worker.process_message.call_count == 1  # 1번만 호출
        assert worker._consumer.commit.call_count == 1  # 커밋 1번
        assert worker._producer.send_and_wait.call_count == 1  # 응답 발행 1번

    @pytest.mark.asyncio
    async def test_deserialization_error_immediate_dlq(
        self, worker: MockWorker
    ) -> None:
        """2. DeserializationError → 즉시 DLQ."""
        # 잘못된 JSON 메시지
        invalid_record = ConsumerRecord(
            topic="test-request",
            partition=0,
            offset=100,
            timestamp=0,
            timestamp_type=0,
            key=None,
            value=b'{"invalid json',  # 깨진 JSON
            checksum=None,
            serialized_key_size=0,
            serialized_value_size=0,
            headers=[],
        )

        # _send_to_dlq Mock
        worker._send_to_dlq = AsyncMock()

        # 실행
        await worker._process_record(invalid_record)

        # 검증
        assert worker._send_to_dlq.call_count == 1  # DLQ 발행 1번
        dlq_call = worker._send_to_dlq.call_args
        assert dlq_call.kwargs["error_type"] == "DeserializationError"
        assert dlq_call.kwargs["retry_count"] == 0  # 재시도 없음
        assert worker._consumer.commit.call_count == 1  # 커밋 1번

    @pytest.mark.asyncio
    async def test_infrastructure_error_retry_then_success(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """3. InfrastructureError → 재시도 2회 → 성공."""
        # process_message Mock 설정: 1차 실패, 2차 성공
        worker.process_message = AsyncMock(
            side_effect=[
                InfrastructureError("Qdrant timeout", service="qdrant"),
                OutfitResponseMessage(
                    request_id="test-123",
                    status="completed",
                    outfits=[],
                ),
            ]
        )

        # asyncio.sleep Mock
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # 실행
            await worker._process_record(mock_record)

            # 검증
            assert worker.process_message.call_count == 2  # 2번 시도
            assert mock_sleep.call_count == 0  # 1차 재시도는 즉시 (0초)
            assert worker._consumer.commit.call_count == 1  # 성공 후 커밋
            assert worker._producer.send_and_wait.call_count == 1  # 응답 발행

    @pytest.mark.asyncio
    async def test_infrastructure_error_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """4. InfrastructureError → 재시도 소진 → DLQ."""
        # process_message Mock 설정: 3회 연속 실패
        worker.process_message = AsyncMock(
            side_effect=[
                InfrastructureError("Qdrant timeout", service="qdrant"),
                InfrastructureError("Qdrant timeout", service="qdrant"),
                InfrastructureError("Qdrant timeout", service="qdrant"),
            ]
        )

        # _send_to_dlq, _send_error_response Mock
        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        # asyncio.sleep Mock
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # 실행
            await worker._process_record(mock_record)

            # 검증
            assert worker.process_message.call_count == 3  # 3번 시도
            assert mock_sleep.call_count == 1  # 2차 재시도만 대기 (5초)
            mock_sleep.assert_called_once_with(5)

            # DLQ 발행 확인
            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "InfrastructureError"
            assert dlq_call.kwargs["retry_count"] == 2  # 재시도 2회

            # ErrorResponse 발행 확인
            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["request_id"] == "test-123"
            assert error_call.kwargs["error_code"] == "INFRASTRUCTURE_ERROR"
            assert error_call.kwargs["retry_after_seconds"] == 30

            # 커밋 확인
            assert worker._consumer.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_retry_then_success(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """5. RateLimitError → retry_after 대기 → 성공."""
        # process_message Mock 설정: 1차 Rate Limit, 2차 성공
        worker.process_message = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
                OutfitResponseMessage(
                    request_id="test-123",
                    status="completed",
                    outfits=[],
                ),
            ]
        )

        # asyncio.sleep Mock
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # 실행
            await worker._process_record(mock_record)

            # 검증
            assert worker.process_message.call_count == 2  # 2번 시도
            assert mock_sleep.call_count == 1  # retry_after 대기
            mock_sleep.assert_called_once_with(3)  # 3초 대기

            assert worker._consumer.commit.call_count == 1  # 성공 후 커밋

    @pytest.mark.asyncio
    async def test_rate_limit_error_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """6. RateLimitError → 재시도 소진 → DLQ."""
        # process_message Mock 설정: 3회 연속 Rate Limit
        worker.process_message = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
            ]
        )

        # _send_to_dlq, _send_error_response Mock
        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        # asyncio.sleep Mock
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # 실행
            await worker._process_record(mock_record)

            # 검증
            assert worker.process_message.call_count == 3  # 3번 시도
            assert mock_sleep.call_count == 2  # 2회 대기 (1차, 2차 재시도)
            # 각 호출이 3초 대기
            assert all(call.args[0] == 3 for call in mock_sleep.call_args_list)

            # DLQ 발행 확인
            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "RateLimitError"
            assert dlq_call.kwargs["retry_count"] == 2

            # ErrorResponse 발행 확인
            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["error_code"] == "RATE_LIMIT_ERROR"
            assert error_call.kwargs["retry_after_seconds"] == 60

    @pytest.mark.asyncio
    async def test_general_exception_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """7. 일반 Exception → 재시도 2회 → DLQ."""
        # process_message Mock 설정: 3회 연속 ValueError
        worker.process_message = AsyncMock(
            side_effect=[
                ValueError("Unexpected error"),
                ValueError("Unexpected error"),
                ValueError("Unexpected error"),
            ]
        )

        # _send_to_dlq, _send_error_response Mock
        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        # asyncio.sleep Mock
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # 실행
            await worker._process_record(mock_record)

            # 검증
            assert worker.process_message.call_count == 3  # 3번 시도
            assert mock_sleep.call_count == 1  # 2차 재시도만 대기 (5초)
            mock_sleep.assert_called_once_with(5)

            # DLQ 발행 확인
            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "ValueError"
            assert dlq_call.kwargs["retry_count"] == 2

            # ErrorResponse 발행 확인
            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["error_code"] == "PROCESSING_ERROR"
            assert error_call.kwargs["retry_after_seconds"] is None

    @pytest.mark.asyncio
    async def test_shutdown_signal_during_retry(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        """보너스: 재시도 중 종료 시그널 수신 시 커밋 없이 종료."""
        # process_message Mock 설정: 계속 실패
        worker.process_message = AsyncMock(
            side_effect=InfrastructureError("Qdrant timeout", service="qdrant")
        )

        # 첫 번째 재시도 후 종료 시그널 설정
        async def set_shutdown_after_first_retry(*args: object) -> None:
            worker._shutdown_event.set()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = set_shutdown_after_first_retry

            # 실행
            await worker._process_record(mock_record)

            # 검증
            # 1차 시도 실패 → 즉시 재시도(0초) → 2차 시도 실패 → sleep(5) 호출 시 종료 시그널 설정 → 루프 재진입 시 종료
            assert worker.process_message.call_count == 2  # 2번 시도 후 종료
            assert worker._consumer.commit.call_count == 0  # 커밋 없음 (Kafka 재전달)
