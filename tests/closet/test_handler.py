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
        return_value=PreprocessResult(
            file_ids=[12345, 12346],
            task_ids=["task-ulid-1", "task-ulid-2"],
            item_images=[b"item1_bytes", b"item2_bytes"],
            success=True,
        )
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
    service.validate_image = AsyncMock(
        return_value={
            "url": "https://example.com/test.jpg",
            "nsfw": {"is_nsfw": False, "score": 0.0},
            "fashion": {"is_fashion": True, "score": 1.0},
            "error": None,
        }
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
                "sourceId": "source-001",
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
    """정상 처리: 어뷰징→세그멘테이션→전처리×N→분석×N→커밋."""
    producer, consumer = mock_kafka

    with (
        patch("app.closet.handler._get_service", return_value=mock_service),
        patch("app.closet.handler.get_kafka_producer", return_value=producer),
    ):
        handler = create_handler(consumer)
        await handler(fake_message)

    # service 호출 검증
    mock_service.preprocess.assert_called_once()
    assert mock_service.analyze.call_count == 2  # 아이템 2개

    # Kafka 결과 토픽에 이벤트 발행:
    # 1 (ABUSING) + 1 (SEGMENTATION) + 2 (PREPROCESSING) + 2 (ANALYZING) = 6
    assert producer.send_and_wait.call_count == 6

    # 오프셋 커밋 1회
    consumer.commit.assert_called_once()


@pytest.mark.asyncio
async def test_handle_analysis_request_preprocess_failure(
    mock_service: MagicMock, mock_kafka: tuple, fake_message: MagicMock
) -> None:
    """전처리 실패: 어뷰징 이벤트만 발행 후 analyze 호출 안 함."""
    producer, consumer = mock_kafka
    mock_service.preprocess.return_value = PreprocessResult(
        success=False, error="SEGMENTATION_FAILED"
    )

    with (
        patch("app.closet.handler._get_service", return_value=mock_service),
        patch("app.closet.handler.get_kafka_producer", return_value=producer),
    ):
        handler = create_handler(consumer)
        await handler(fake_message)

    # 전처리 실패 시 analyze는 호출되면 안 됨
    mock_service.preprocess.assert_called_once()
    mock_service.analyze.assert_not_called()

    # 어뷰징 이벤트 1건만 발행, 커밋 1회
    assert producer.send_and_wait.call_count == 1
    consumer.commit.assert_called_once()
