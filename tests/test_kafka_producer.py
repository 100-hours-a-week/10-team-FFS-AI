"""테스트용 Kafka Producer — 가짜 분석 요청 메시지를 요청 토픽에 보냅니다.

사용법:
    python tests/test_kafka_producer.py

이 스크립트가 하는 일:
    1. Kafka 요청 토픽(ai.clothes.analyze.request)에 연결
    2. AI_ANALYSIS_REQUESTED 이벤트 메시지 1건 전송
    3. 백엔드가 하는 일을 우리가 직접 흉내내는 것!
"""

import asyncio
import json
import uuid
from datetime import UTC, datetime

from aiokafka import AIOKafkaProducer


async def main() -> None:
    bootstrap_servers = "localhost:9092"
    request_topic = "ai.clothes.analyze.request"

    # S3 presigned URL (1시간 유효)
    target_image = (
        "https://my-vton-test-bucket-2025.s3.amazonaws.com/"
        "test_validation/test_fashion1.png"
        "?AWSAccessKeyId=AKIAQPZU5KQL3QNZXR5U"
        "&Signature=2axBnmXEh%2F5vaPzqqZp5F3yazxw%3D"
        "&Expires=1772022704"
    )

    # 백엔드가 보내는 것과 동일한 형태의 메시지
    test_message = {
        "eventType": "AI_ANALYSIS_REQUESTED",
        "requestedAt": datetime.now(UTC).isoformat(),
        "data": {
            "batchId": str(uuid.uuid4()),
            "taskId": str(uuid.uuid4()),
            "userId": 1,
            "targetImage": target_image,
        },
    }

    print("=" * 60)
    print("Kafka 테스트 Producer")
    print("=" * 60)
    print(f"토픽: {request_topic}")
    print(f"batchId: {test_message['data']['batchId']}")
    print(f"taskId: {test_message['data']['taskId']}")
    print(f"이미지: {target_image[:80]}...")
    print("=" * 60)

    # Producer 생성 및 메시지 전송
    producer = AIOKafkaProducer(bootstrap_servers=bootstrap_servers)
    await producer.start()

    try:
        message_bytes = json.dumps(test_message).encode("utf-8")
        result = await producer.send_and_wait(request_topic, message_bytes)
        print("\n✅ 메시지 전송 성공!")
        print(f"   파티션: {result.partition}")
        print(f"   오프셋: {result.offset}")
        print("\n이제 AI 서버(uvicorn)를 실행하면 이 메시지를 받아서 처리합니다.")
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
