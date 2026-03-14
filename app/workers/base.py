from __future__ import annotations

import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord

from app.common.kafka.exceptions import (
    DeserializationError,
    InfrastructureError,
    PermanentError,
    RateLimitError,
    RetryableError,
)
from app.common.kafka.schemas import (
    DLQErrorInfo,
    DLQMessage,
    ErrorDetail,
    ErrorResponse,
    ProgressMessage,
)
from app.common.kafka.serialization import deserialize, serialize
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

    async def _process_record_once(self, record: ConsumerRecord) -> None:
        """단일 메시지 처리 (재시도 없음).

        역직렬화, 비즈니스 로직 처리, 응답 발행, 커밋을 수행합니다.
        예외가 발생하면 호출자(_process_record)에서 처리합니다.

        Args:
            record: Kafka ConsumerRecord

        Raises:
            DeserializationError: 역직렬화 실패 시
            Exception: 메시지 처리 중 발생한 모든 예외
        """
        start_time = time.time()

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

    async def _process_record(self, record: ConsumerRecord) -> None:
        """메시지 처리 진입점 (재시도 로직 포함).

        _process_record_once를 호출하여 메시지를 처리하고,
        예외 타입에 따라 재시도하거나 DLQ로 전송합니다.

        Args:
            record: Kafka ConsumerRecord
        """
        retry_count = 0
        max_retries = 2
        last_error: Exception | None = None

        while retry_count <= max_retries:
            # 종료 시그널 확인
            if self._shutdown_event.is_set():
                logger.info("재시도 중 종료 시그널 수신, 커밋하지 않고 종료")
                return  # 커밋 없이 종료 → Kafka가 재전달

            try:
                await self._process_record_once(record)
                return  # 성공 시 즉시 종료

            except PermanentError as e:
                # 영구적 실패 → 재시도 없이 즉시 DLQ
                logger.error(f"영구적 실패 발생: {e.__class__.__name__}: {e}")
                await self._handle_permanent_failure(record, e)
                return

            except RateLimitError as e:
                # Rate Limit → retry_after 대기 (상한: 60초)
                if retry_count < max_retries:
                    actual_wait = min(e.retry_after, 60)
                    if actual_wait < e.retry_after:
                        logger.warning(
                            f"retry_after={e.retry_after}초는 너무 김, "
                            f"{actual_wait}초로 제한"
                        )
                    logger.warning(
                        f"Rate limit 발생: service={e.service}, "
                        f"retry_after={actual_wait}초, retry_count={retry_count}"
                    )
                    await asyncio.sleep(actual_wait)
                    retry_count += 1
                    last_error = e
                else:
                    # 재시도 소진
                    logger.error(
                        f"재시도 소진 (Rate Limit): retry_count={retry_count}, "
                        f"service={e.service}"
                    )
                    await self._handle_retry_exhausted(record, e, retry_count)
                    return

            except RetryableError as e:
                # 일시적 실패 → 백오프 후 재시도
                if retry_count < max_retries:
                    backoff_seconds = 0 if retry_count == 0 else 5
                    logger.warning(
                        f"재시도 가능한 오류 발생: {e.__class__.__name__}, "
                        f"retry_count={retry_count}, backoff={backoff_seconds}초"
                    )
                    if backoff_seconds > 0:
                        await asyncio.sleep(backoff_seconds)
                    retry_count += 1
                    last_error = e
                else:
                    # 재시도 소진
                    logger.error(
                        f"재시도 소진: retry_count={retry_count}, "
                        f"error={e.__class__.__name__}: {e}"
                    )
                    await self._handle_retry_exhausted(record, e, retry_count)
                    return

            except Exception as e:
                # 알 수 없는 예외 → 안전하게 재시도
                if retry_count < max_retries:
                    backoff_seconds = 0 if retry_count == 0 else 5
                    logger.error(
                        f"예상치 못한 예외 발생: {e.__class__.__name__}: {e}, "
                        f"retry_count={retry_count}, backoff={backoff_seconds}초"
                    )
                    if backoff_seconds > 0:
                        await asyncio.sleep(backoff_seconds)
                    retry_count += 1
                    last_error = e
                else:
                    logger.error(
                        f"재시도 소진: retry_count={retry_count}, "
                        f"last_error={last_error.__class__.__name__}: {last_error}"
                    )
                    await self._handle_retry_exhausted(record, last_error, retry_count)
                    return

    async def _handle_permanent_failure(
        self,
        record: ConsumerRecord,
        error: PermanentError,
    ) -> None:
        """영구적 실패 처리 (즉시 DLQ).

        Args:
            record: Kafka ConsumerRecord
            error: 발생한 PermanentError
        """
        original_message = record.value.decode("utf-8", errors="replace")

        await self._send_to_dlq(
            original_message=original_message,
            error_type=error.__class__.__name__,
            error_message=str(error),
            stack_trace=self._get_stack_trace(error),
            retry_count=0,
        )

        if self._consumer:
            await self._consumer.commit()

    async def _handle_retry_exhausted(
        self,
        record: ConsumerRecord,
        last_error: Exception,
        retry_count: int,
    ) -> None:
        """재시도 소진 후 DLQ 처리.

        Args:
            record: Kafka ConsumerRecord
            last_error: 마지막 발생한 예외
            retry_count: 재시도 횟수
        """
        # request_id 추출 시도 (에러 응답 발행용)
        request_id: str | None = None
        try:
            message = deserialize(record.value, self.request_model_class)
            request_id = getattr(message, "request_id", None)
        except Exception:
            pass

        # DLQ 발행
        original_message = record.value.decode("utf-8", errors="replace")
        await self._send_to_dlq(
            original_message=original_message,
            error_type=last_error.__class__.__name__,
            error_message=str(last_error),
            stack_trace=self._get_stack_trace(last_error),
            retry_count=retry_count,
        )

        # ErrorResponse 발행 (request_id가 있을 때만)
        if request_id:
            error_code = self._map_error_code(last_error)
            retry_after = self._get_retry_after(last_error)
            await self._send_error_response(
                request_id=request_id,
                error_code=error_code,
                error_message=str(last_error),
                retry_after_seconds=retry_after,
            )

        # 커밋
        if self._consumer:
            await self._consumer.commit()

    def _get_stack_trace(self, error: Exception) -> str:
        """예외에서 스택 트레이스 추출.

        Args:
            error: 예외 객체

        Returns:
            스택 트레이스 문자열
        """
        import traceback

        return "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )

    def _map_error_code(self, error: Exception) -> str:
        """예외 타입을 에러 코드로 매핑.

        Args:
            error: 예외 객체

        Returns:
            에러 코드 문자열
        """
        if isinstance(error, DeserializationError):
            return "DESERIALIZATION_ERROR"
        elif isinstance(error, RateLimitError):
            return "RATE_LIMIT_ERROR"
        elif isinstance(error, InfrastructureError):
            return "INFRASTRUCTURE_ERROR"
        else:
            return "PROCESSING_ERROR"

    def _get_retry_after(self, error: Exception) -> int | None:
        """예외 타입별 재시도 권장 대기 시간.

        Args:
            error: 예외 객체

        Returns:
            재시도 권장 대기 시간 (초), 없으면 None
        """
        if isinstance(error, RateLimitError):
            return 60
        elif isinstance(error, InfrastructureError):
            return 30
        else:
            return None

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
