import asyncio
import logging
import signal
import time
from abc import ABC, abstractmethod

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord

from app.common.kafka.schemas import ProgressMessage
from app.common.kafka.serialization import serialize

logger = logging.getLogger(__name__)


class BaseWorker(ABC):
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        consume_topic: str,
        produce_topic: str,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consume_topic = consume_topic
        self.produce_topic = produce_topic

        self.consumer = None
        self.producer = None
        self.is_running = True

    async def setup(self) -> None:
        self.consumer = AIOKafkaConsumer(
            self.consume_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_records=1,
        )
        self.producer = AIOKafkaProducer(bootstrap_servers=self.bootstrap_servers)

        await self.consumer.start()
        await self.producer.start()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_exit_signal)

    def _handle_exit_signal(self) -> None:
        print(
            "\n[BaseWorker] 종료 신호 수신. 현재 메시지 처리 후 안전하게 종료합니다..."
        )
        self.is_running = False

    async def teardown(self) -> None:
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()

    async def send_progress(self, request_id: str, step: str, step_label: str) -> None:
        progress = ProgressMessage(
            request_id=request_id,
            status="processing",
            step=step,
            step_label=step_label,
            timestamp=time.time(),
        )

        await self.producer.send_and_wait(self.produce_topic, serialize(progress))

    @abstractmethod
    async def process_message(self, message: ConsumerRecord) -> None:
        pass

    @abstractmethod
    async def _handle_failure(self, message: ConsumerRecord, error: Exception) -> None:
        pass

    async def run(self) -> None:
        await self.setup()

        try:
            print(f"[BaseWorker] {self.consume_topic} 구독 시작...")
            while self.is_running:
                msg_set = await self.consumer.getmany(timeout_ms=1000)

                for _topic_partition, messages in msg_set.items():
                    for msg in messages:
                        try:
                            await self.process_message(msg)

                        except Exception as e:
                            logger.error(
                                f"메시지 처리 실패 (offset: {msg.offset}): {e}"
                            )
                            await self._handle_failure(msg, e)

                        finally:
                            await self.consumer.commit()

        finally:
            await self.teardown()
