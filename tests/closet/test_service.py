"""ClosetService 단위 테스트 — 비즈니스 로직(이미지 분석 파이프라인) 검증

외부 의존(S3, Gemini, Backend API, Segmentation)을 모두 Mock으로 대체하여
순수 비즈니스 로직만 테스트합니다.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.service import ClosetService
from app.common.llm_schemas import (
    ImageAnalysisResult,
    ImageExtraAttributes,
    ImageExtraMetadata,
    ImageMajorAttributes,
)


@pytest.fixture
def mock_s3_client() -> MagicMock:
    client = MagicMock()
    client.get_image = AsyncMock(return_value=b"fake_image_bytes")
    client.put_image = AsyncMock()
    return client


@pytest.fixture
def mock_analyzer() -> MagicMock:
    analyzer = MagicMock()
    analyzer.analyze_image = AsyncMock(
        return_value=ImageAnalysisResult(
            major=ImageMajorAttributes(
                category="TOP",
                color=["흰색"],
                material=["면"],
                style_tags=["캐주얼"],
            ),
            extra=ImageExtraAttributes(
                meta_data=ImageExtraMetadata(
                    gender="유니섹스",
                    season=["봄", "가을"],
                    formality="캐주얼",
                    fit="레귤러핏",
                    occasion=["데일리"],
                ),
                caption="흰색 면 티셔츠입니다.",
            ),
        )
    )
    return analyzer


@pytest.fixture
def mock_segmentation() -> MagicMock:
    seg = MagicMock()
    seg.segment = AsyncMock(return_value=[b"item1_bytes", b"item2_bytes"])
    return seg


@pytest.fixture
def service(
    mock_s3_client: MagicMock, mock_analyzer: MagicMock, mock_segmentation: MagicMock
) -> Generator[ClosetService, Any, None]:
    with (
        patch("app.closet.service.GeminiImageAnalyzer"),
        patch(
            "app.closet.segmentation.SegmentationService",
            return_value=mock_segmentation,
        ),
    ):
        svc = ClosetService()
    svc._s3_client = mock_s3_client
    svc._analyzer = mock_analyzer
    svc._segmentation = mock_segmentation
    yield svc


# ── preprocess 테스트 ──


@pytest.mark.asyncio
async def test_preprocess_success(
    service: ClosetService, mock_segmentation: MagicMock
) -> None:
    """전처리 성공: 세그멘테이션 → presigned URL 발급 → S3 업로드."""
    with patch.object(
        service,
        "_request_presigned_urls",
        new_callable=AsyncMock,
        return_value=[
            {"fileId": 101, "presignedUrl": "https://s3.example.com/put1"},
            {"fileId": 102, "presignedUrl": "https://s3.example.com/put2"},
        ],
    ):
        result = await service.preprocess(
            target_image_url="https://example.com/image.jpg",
            user_id=1,
        )

    assert result.success is True
    assert result.file_ids == [101, 102]
    assert len(result.task_ids) == 2
    assert len(result.item_images) == 2
    assert result.error is None
    mock_segmentation.segment.assert_called_once_with("https://example.com/image.jpg")


@pytest.mark.asyncio
async def test_preprocess_segmentation_failure(
    service: ClosetService, mock_segmentation: MagicMock
) -> None:
    """전처리 실패: 세그멘테이션 실패 시 에러 반환."""
    mock_segmentation.segment.side_effect = Exception("Segmentation failed")

    result = await service.preprocess(
        target_image_url="https://example.com/bad.jpg",
        user_id=1,
    )

    assert result.success is False
    assert result.error == "SEGMENTATION_FAILED"


@pytest.mark.asyncio
async def test_preprocess_upload_failure(
    service: ClosetService, mock_s3_client: MagicMock, mock_segmentation: MagicMock
) -> None:
    """전처리 실패: 모든 S3 업로드 실패 시 에러 반환."""
    mock_s3_client.put_image.side_effect = Exception("Upload failed")

    with patch.object(
        service,
        "_request_presigned_urls",
        new_callable=AsyncMock,
        return_value=[
            {"fileId": 101, "presignedUrl": "https://s3.example.com/put1"},
            {"fileId": 102, "presignedUrl": "https://s3.example.com/put2"},
        ],
    ):
        result = await service.preprocess(
            target_image_url="https://example.com/image.jpg",
            user_id=1,
        )

    assert result.success is False
    assert "ALL_UPLOADS_FAILED" in result.error


# ── analyze 테스트 ──


@pytest.mark.asyncio
async def test_analyze_success(
    service: ClosetService, mock_analyzer: MagicMock
) -> None:
    """분석 성공: bytes를 직접 받아 분석."""
    result = await service.analyze(image_bytes=b"fake_item_bytes")

    assert result.success is True
    assert result.major.category == "TOP"
    assert result.major.color == ["흰색"]
    assert result.extra.caption == "흰색 면 티셔츠입니다."
    mock_analyzer.analyze_image.assert_called_once_with(b"fake_item_bytes")


@pytest.mark.asyncio
async def test_analyze_vlm_failure_fallback(
    service: ClosetService, mock_analyzer: MagicMock
) -> None:
    """분석 실패: VLM 에러 시 기본값(DEFAULT_ANALYSIS) 사용."""
    mock_analyzer.analyze_image.side_effect = Exception("Gemini API error")

    result = await service.analyze(image_bytes=b"fake_item_bytes")

    # VLM이 에러나도 기본값으로 결과가 반환되어야 함
    assert result.success is True
    assert result.major.category == "ETC"
