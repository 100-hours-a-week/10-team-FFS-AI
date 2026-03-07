import asyncio
import signal
import time
from abc import ABC, abstractmethod
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from app.common.kafka.serialization import serialize
from app.common.kafka.schemas import ProgressMessage


class BaseWorker(ABC):
    def __init__(
            self,
            bootstrap_servers: str,
            group_id: str,
            consume_topic: str,
            produce_topic: str
    ):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consume_topic = consume_topic
        self.produce_topic = produce_topic

        self.consumer = None
        self.producer = None
        self.is_running = True

    async def setup(self):
        """Kafka Consumer와 Producer를 초기화합니다."""
        # V2 설계에 따라 max_poll_records=1, 수동 커밋 모드로 설정합니다.
        self.consumer = AIOKafkaConsumer(
            self.consume_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            max_poll_records=1
        )
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers
        )

        await self.consumer.start()
        await self.producer.start()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """SIGTERM, SIGINT 신호를 감지하여 Graceful Shutdown을 준비합니다."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._handle_exit_signal)

    def _handle_exit_signal(self):
        """종료 신호 수신 시 플래그를 변경합니다."""
        print(f"\n[BaseWorker] 종료 신호 수신. 현재 메시지 처리 후 안전하게 종료합니다...")
        self.is_running = False

    async def teardown(self):
        """자원을 안전하게 해제합니다."""
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()

    async def send_progress(self, request_id: str, step: str, step_label: str):
        """작업 진행 상태를 발행합니다."""
        progress = ProgressMessage(
            request_id=request_id,
            status="processing",
            step=step,
            step_label=step_label,
            timestamp=time.time()
        )

        await self.producer.send_and_wait(
            self.produce_topic,
            serialize(progress)
        )

    @abstractmethod
    async def process_message(self, message):
        """하위 클래스에서 실제 비즈니스 로직을 구현합니다."""
        pass

    async def run(self):
        """메인 메시지 수신 루프입니다."""
        await self.setup()

        try:
            print(f"[BaseWorker] {self.consume_topic} 구독 시작...")
            while self.is_running:
                # max_poll_records=1 설정으로 한 번에 하나의 메시지만 가져옵니다.
                msg_set = await self.consumer.getmany(timeout_ms=1000)

                for topic_partition, messages in msg_set.items():
                    for msg in messages:
                        await self.process_message(msg)
                        # 처리가 완료된 후 수동으로 커밋하여 유실을 방지합니다.
                        await self.consumer.commit()

        finally:
            await self.teardown()