"""handler 단위 테스트 — Kafka 메시지 처리 오케스트레이션 검증

외부 의존(Kafka, ClosetService)을 모두 Mock으로 대체하여
handler가 올바른 순서로 이벤트를 발행하고 오프셋을 커밋하는지 테스트합니다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.handler import create_handler
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.closet.service import AnalysisResult, PreprocessResult


@pytest.fixture
def mock_service() -> MagicMock:
    service = MagicMock()
    service.preprocess = AsyncMock(
        return_value=PreprocessResult(file_id=12345, success=True)
    )
    service.analyze = AsyncMock(
        return_value=AnalysisResult(
            major=MajorAttributes(
                category="TOP",
                color=["흰색"],
                material=["면"],
                style_tags=["캐주얼"],
            ),
            extra=ExtraAttributes(
                meta_data=ExtraMetadata(
                    gender="유니섹스",
                    season=["봄"],
                    formality="캐주얼",
                    fit="레귤러핏",
                ),
                caption="흰색 면 티셔츠입니다.",
            ),
            success=True,
        )
    )
    return service


@pytest.fixture
def mock_kafka() -> tuple[AsyncMock, AsyncMock]:
    producer = AsyncMock()
    consumer = AsyncMock()
    return producer, consumer


@pytest.fixture
def fake_message() -> MagicMock:
    """백엔드가 보내는 것과 동일한 형태의 Kafka ConsumerRecord Mock."""
    import json

    raw = json.dumps(
        {
            "eventType": "AI_ANALYSIS_REQUESTED",
            "requestedAt": "2026-02-26T06:28:02.000000+00:00",
            "data": {
                "batchId": "batch-001",
                "taskId": "task-001",
                "userId": 1,
                "targetImage": "https://example.com/test.jpg",
            },
        }
    ).encode("utf-8")

    message = MagicMock()
    message.value = raw
    return message


@pytest.mark.asyncio
async def test_handle_analysis_request_success(
    mock_service: MagicMock, mock_kafka: tuple, fake_message: MagicMock
) -> None:
    """정상 처리: preprocess → 이벤트 발행 → analyze → 이벤트 발행 → 커밋."""
    producer, consumer = mock_kafka

    with (
        patch("app.closet.handler._service", mock_service),
        patch("app.closet.handler.get_kafka_producer", return_value=producer),
    ):
        handler = create_handler(consumer)
        await handler(fake_message)

    # service 호출 검증
    mock_service.preprocess.assert_called_once()
    mock_service.analyze.assert_called_once()

    # Kafka 결과 토픽에 이벤트 2건 발행 (PREPROCESSING + ANALYZING)
    assert producer.send_and_wait.call_count == 2

    # 오프셋 커밋 1회
    consumer.commit.assert_called_once()


@pytest.mark.asyncio
async def test_handle_analysis_request_preprocess_failure(
    mock_service: MagicMock, mock_kafka: tuple, fake_message: MagicMock
) -> None:
    """전처리 실패: analyze를 호출하지 않고 커밋만 수행."""
    producer, consumer = mock_kafka
    mock_service.preprocess.return_value = PreprocessResult(
        file_id=0, success=False, error="IMAGE_DOWNLOAD_FAILED"
    )

    with (
        patch("app.closet.handler._service", mock_service),
        patch("app.closet.handler.get_kafka_producer", return_value=producer),
    ):
        handler = create_handler(consumer)
        await handler(fake_message)

    # 전처리 실패 시 analyze는 호출되면 안 됨
    mock_service.preprocess.assert_called_once()
    mock_service.analyze.assert_not_called()

    # 이벤트 발행 0건, 커밋 1회 (실패한 메시지도 커밋해야 무한 재시도 방지)
    producer.send_and_wait.assert_not_called()
    consumer.commit.assert_called_once()
