from __future__ import annotations

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord

from app.common.kafka.schemas import (
    DLQErrorInfo,
    DLQMessage,
    ErrorDetail,
    ErrorResponse,
    ProgressMessage,
)
from app.common.kafka.serialization import (
    DeserializationError,
    deserialize,
    serialize,
)
from app.common.schemas import BaseSchema
from app.workers.config import WorkerConfig

logger = logging.getLogger(__name__)


TRequest = TypeVar("TRequest", bound=BaseSchema)


class BaseWorker(ABC, Generic[TRequest]):
    request_model_class: type[TRequest]

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._consumer: AIOKafkaConsumer | None = None
        self._producer: AIOKafkaProducer | None = None
        self._shutdown_event = asyncio.Event()
        self._running = False

    @property
    @abstractmethod
    def request_topic(self) -> str: ...

    @property
    @abstractmethod
    def response_topic(self) -> str: ...

    @property
    @abstractmethod
    def dlq_topic(self) -> str: ...

    @property
    @abstractmethod
    def group_id(self) -> str: ...

    @abstractmethod
    async def process_message(self, message: TRequest) -> BaseSchema: ...

    async def start(self) -> None:
        logger.info(f"Worker 시작: topic={self.request_topic}, group={self.group_id}")

        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_shutdown_signal)

        self._producer = AIOKafkaProducer(
            bootstrap_servers=self.config.kafka_bootstrap_servers,
        )
        await self._producer.start()
        logger.info("Kafka Producer 시작 완료")

        self._consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.config.kafka_bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset=self.config.consumer_auto_offset_reset,
            enable_auto_commit=False,
            max_poll_records=self.config.consumer_max_poll_records,
            session_timeout_ms=self.config.consumer_session_timeout_ms,
            heartbeat_interval_ms=self.config.consumer_heartbeat_interval_ms,
        )
        await self._consumer.start()
        logger.info("Kafka Consumer 시작 완료")

        self._running = True

    async def stop(self) -> None:
        logger.info("Worker 종료 시작...")
        self._running = False

        if self._consumer:
            try:
                await self._consumer.stop()
                logger.info("Kafka Consumer 종료 완료")
            except Exception as e:
                logger.error(f"Consumer 종료 실패: {e}")

        if self._producer:
            try:
                await self._producer.stop()
                logger.info("Kafka Producer 종료 완료")
            except Exception as e:
                logger.error(f"Producer 종료 실패: {e}")

        logger.info("Worker 종료 완료")

    def _handle_shutdown_signal(self) -> None:
        logger.info("종료 시그널 수신, graceful shutdown 시작...")
        self._shutdown_event.set()

    async def run(self) -> None:
        if not self._consumer or not self._producer:
            raise RuntimeError(
                "Worker가 시작되지 않았습니다. start()를 먼저 호출하세요."
            )

        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)

        logger.info(
            f"Consumer 루프 시작 "
            f"(max_concurrent_tasks={self.config.max_concurrent_tasks})"
        )

        try:
            async for record in self._consumer:
                if self._shutdown_event.is_set():
                    logger.info("종료 시그널 감지, 루프 종료")
                    break

                await semaphore.acquire()
                asyncio.create_task(
                    self._process_record_with_release(record, semaphore)
                )

        except asyncio.CancelledError:
            logger.info("Consumer 루프 취소됨")
        except Exception as e:
            logger.error(f"Consumer 루프 예외: {e}")
            raise

    async def _process_record_with_release(
        self,
        record: ConsumerRecord,
        semaphore: asyncio.Semaphore,
    ) -> None:
        try:
            await self._process_record(record)
        finally:
            semaphore.release()

    async def _process_record(self, record: ConsumerRecord) -> None:
        start_time = time.time()
        request_id: str | None = None

        try:
            message = deserialize(record.value, self.request_model_class)
            request_id = getattr(message, "request_id", None)

            logger.info(
                f"메시지 수신: request_id={request_id}, "
                f"offset={record.offset}, partition={record.partition}"
            )

            response = await self.process_message(message)

            await self._send_response(response, key=request_id)

            if self._consumer:
                await self._consumer.commit()

            elapsed_ms = int((time.time() - start_time) * 1000)
            logger.info(
                f"메시지 처리 완료: request_id={request_id}, elapsed_ms={elapsed_ms}"
            )

        except DeserializationError as e:
            logger.error(f"역직렬화 실패: {e}")
            await self._send_to_dlq(
                original_message=e.original_data.decode("utf-8", errors="replace"),
                error_type="DeserializationError",
                error_message=str(e),
            )
            if self._consumer:
                await self._consumer.commit()

        except Exception as e:
            logger.error(f"메시지 처리 실패: request_id={request_id}, error={e}")
            if request_id:
                await self._send_error_response(
                    request_id=request_id,
                    error_code="PROCESSING_ERROR",
                    error_message=str(e),
                )
            if self._consumer:
                await self._consumer.commit()

    async def send_progress(
        self,
        request_id: str,
        step: int,
        step_label: str,
    ) -> None:
        progress = ProgressMessage(
            request_id=request_id,
            step=step,
            step_label=step_label,
        )
        await self._send_response(progress, key=request_id)
        logger.debug(f"Progress 발행: request_id={request_id}, step={step}")

    async def _send_response(
        self,
        message: BaseSchema,
        key: str | None = None,
    ) -> None:
        if not self._producer:
            raise RuntimeError("Producer가 초기화되지 않았습니다.")

        key_bytes = key.encode("utf-8") if key else None
        await self._producer.send_and_wait(
            self.response_topic,
            serialize(message),
            key=key_bytes,
        )

    async def _send_error_response(
        self,
        request_id: str,
        error_code: str,
        error_message: str,
        retry_after_seconds: int | None = None,
    ) -> None:
        error_response = ErrorResponse(
            request_id=request_id,
            error=ErrorDetail(
                code=error_code,
                message=error_message,
                retry_after_seconds=retry_after_seconds,
            ),
        )
        await self._send_response(error_response, key=request_id)
        logger.info(f"에러 응답 발행: request_id={request_id}, code={error_code}")

    async def _send_to_dlq(
        self,
        original_message: str,
        error_type: str,
        error_message: str,
        stack_trace: str | None = None,
        retry_count: int = 0,
    ) -> None:
        if not self._producer:
            raise RuntimeError("Producer가 초기화되지 않았습니다.")

        dlq_message = DLQMessage(
            original_topic=self.request_topic,
            original_message=original_message,
            error=DLQErrorInfo(
                type=error_type,
                message=error_message,
                stack_trace=stack_trace,
            ),
            retry_count=retry_count,
        )
        await self._producer.send_and_wait(
            self.dlq_topic,
            serialize(dlq_message),
        )
        logger.info(f"DLQ 발행 완료: topic={self.dlq_topic}")
