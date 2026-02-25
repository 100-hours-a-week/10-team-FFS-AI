"""테스트용 Kafka Consumer — 결과 토픽을 구독해서 AI 서버의 응답을 확인합니다.

사용법:
    python tests/test_kafka_consumer.py

이 스크립트가 하는 일:
    1. Kafka 결과 토픽(ai.clothes.analyze.result)에 연결
    2. AI 서버가 보낸 이벤트(PREPROCESSING_COMPLETED, ANALYZING_COMPLETED)를 수신
    3. 백엔드가 받는 것을 우리가 직접 확인하는 것!
"""

import asyncio
import json

from aiokafka import AIOKafkaConsumer


async def main() -> None:
    bootstrap_servers = "localhost:9092"
    result_topic = "ai.clothes.analyze.result"

    print("=" * 60)
    print("Kafka 테스트 Consumer — 결과 토픽 모니터링")
    print(f"토픽: {result_topic}")
    print("메시지를 기다리는 중... (Ctrl+C로 종료)")
    print("=" * 60)

    consumer = AIOKafkaConsumer(
        result_topic,
        bootstrap_servers=bootstrap_servers,
        group_id="test_result_monitor",
        auto_offset_reset="earliest",
    )
    await consumer.start()

    try:
        async for message in consumer:
            data = json.loads(message.value.decode("utf-8"))
            event_type = data.get("eventType", "UNKNOWN")

            print(f"\n{'─' * 60}")
            print(f"📨 이벤트 수신: {event_type}")
            print(f"   파티션: {message.partition}, 오프셋: {message.offset}")
            print("   내용:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
    except asyncio.CancelledError:
        pass
    finally:
        await consumer.stop()
        print("\n종료됨")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n종료됨")
