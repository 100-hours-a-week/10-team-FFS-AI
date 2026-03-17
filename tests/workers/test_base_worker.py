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
    return WorkerConfig()


@pytest.fixture
def worker(worker_config: WorkerConfig) -> MockWorker:
    worker = MockWorker(worker_config)

    worker._consumer = AsyncMock()
    worker._producer = AsyncMock()
    return worker


@pytest.fixture
def mock_record() -> ConsumerRecord:
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

        await worker._process_record(mock_record)

        assert worker.process_message.call_count == 1
        assert worker._consumer.commit.call_count == 1
        assert worker._producer.send_and_wait.call_count == 1

    @pytest.mark.asyncio
    async def test_deserialization_error_immediate_dlq(
        self, worker: MockWorker
    ) -> None:
        invalid_record = ConsumerRecord(
            topic="test-request",
            partition=0,
            offset=100,
            timestamp=0,
            timestamp_type=0,
            key=None,
            value=b'{"invalid json',
            checksum=None,
            serialized_key_size=0,
            serialized_value_size=0,
            headers=[],
        )

        worker._send_to_dlq = AsyncMock()

        await worker._process_record(invalid_record)

        assert worker._send_to_dlq.call_count == 1
        dlq_call = worker._send_to_dlq.call_args
        assert dlq_call.kwargs["error_type"] == "DeserializationError"
        assert dlq_call.kwargs["retry_count"] == 0
        assert worker._consumer.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_infrastructure_error_retry_then_success(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
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

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 2
            assert mock_sleep.call_count == 0
            assert worker._consumer.commit.call_count == 1
            assert worker._producer.send_and_wait.call_count == 1

    @pytest.mark.asyncio
    async def test_infrastructure_error_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        worker.process_message = AsyncMock(
            side_effect=[
                InfrastructureError("Qdrant timeout", service="qdrant"),
                InfrastructureError("Qdrant timeout", service="qdrant"),
                InfrastructureError("Qdrant timeout", service="qdrant"),
            ]
        )

        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 3
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_once_with(5)

            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "InfrastructureError"
            assert dlq_call.kwargs["retry_count"] == 2

            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["request_id"] == "test-123"
            assert error_call.kwargs["error_code"] == "INFRASTRUCTURE_ERROR"
            assert error_call.kwargs["retry_after_seconds"] == 30

            assert worker._consumer.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_retry_then_success(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
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

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 2
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_once_with(3)

            assert worker._consumer.commit.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limit_error_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        worker.process_message = AsyncMock(
            side_effect=[
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
                RateLimitError("Rate limit exceeded", retry_after=3, service="openai"),
            ]
        )

        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 3
            assert mock_sleep.call_count == 2

            assert all(call.args[0] == 3 for call in mock_sleep.call_args_list)

            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "RateLimitError"
            assert dlq_call.kwargs["retry_count"] == 2

            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["error_code"] == "RATE_LIMIT_ERROR"
            assert error_call.kwargs["retry_after_seconds"] == 60

    @pytest.mark.asyncio
    async def test_general_exception_retry_exhausted(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        worker.process_message = AsyncMock(
            side_effect=[
                ValueError("Unexpected error"),
                ValueError("Unexpected error"),
                ValueError("Unexpected error"),
            ]
        )

        worker._send_to_dlq = AsyncMock()
        worker._send_error_response = AsyncMock()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 3
            assert mock_sleep.call_count == 1
            mock_sleep.assert_called_once_with(5)

            assert worker._send_to_dlq.call_count == 1
            dlq_call = worker._send_to_dlq.call_args
            assert dlq_call.kwargs["error_type"] == "ValueError"
            assert dlq_call.kwargs["retry_count"] == 2

            assert worker._send_error_response.call_count == 1
            error_call = worker._send_error_response.call_args
            assert error_call.kwargs["error_code"] == "PROCESSING_ERROR"
            assert error_call.kwargs["retry_after_seconds"] is None

    @pytest.mark.asyncio
    async def test_shutdown_signal_during_retry(
        self, worker: MockWorker, mock_record: ConsumerRecord
    ) -> None:
        worker.process_message = AsyncMock(
            side_effect=InfrastructureError("Qdrant timeout", service="qdrant")
        )

        async def set_shutdown_after_first_retry(*args: object) -> None:
            worker._shutdown_event.set()

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = set_shutdown_after_first_retry

            await worker._process_record(mock_record)

            assert worker.process_message.call_count == 2
            assert worker._consumer.commit.call_count == 0
