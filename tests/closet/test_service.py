"""ClosetService 단위 테스트 — 비즈니스 로직(이미지 분석 파이프라인) 검증

외부 의존(S3, Gemini, Backend API)을 모두 Mock으로 대체하여
순수 비즈니스 로직만 테스트합니다.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.closet.service import ClosetService


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
        return_value={
            "major": {
                "category": "TOP",
                "color": ["흰색"],
                "material": ["면"],
                "style_tags": ["캐주얼"],
            },
            "extra": {
                "meta_data": {
                    "gender": "유니섹스",
                    "season": ["봄", "가을"],
                    "formality": "캐주얼",
                    "fit": "레귤러핏",
                    "occasion": ["데일리"],
                },
                "caption": "흰색 면 티셔츠입니다.",
            },
        }
    )
    return analyzer


@pytest.fixture
def service(
    mock_s3_client: MagicMock, mock_analyzer: MagicMock
) -> Generator[ClosetService, Any, None]:
    with patch("app.closet.service.GeminiImageAnalyzer"):
        svc = ClosetService()
    svc._s3_client = mock_s3_client
    svc._analyzer = mock_analyzer
    yield svc


# ── preprocess 테스트 ──


@pytest.mark.asyncio
async def test_preprocess_success(
    service: ClosetService, mock_s3_client: MagicMock
) -> None:
    """전처리 성공: 이미지 다운로드 + presigned URL 발급 + S3 업로드."""
    with patch.object(
        service,
        "_request_presigned_url",
        new_callable=AsyncMock,
        return_value={"fileId": 12345, "presignedUrl": "https://s3.example.com/put"},
    ):
        result = await service.preprocess(
            target_image_url="https://example.com/image.jpg",
            user_id=1,
        )

    assert result.success is True
    assert result.file_id == 12345
    assert result.error is None
    mock_s3_client.get_image.assert_called_once_with("https://example.com/image.jpg")
    mock_s3_client.put_image.assert_called_once()


@pytest.mark.asyncio
async def test_preprocess_download_failure(
    service: ClosetService, mock_s3_client: MagicMock
) -> None:
    """전처리 실패: 이미지 다운로드 실패 시 에러 반환."""
    mock_s3_client.get_image.side_effect = Exception("Download failed")

    result = await service.preprocess(
        target_image_url="https://example.com/bad.jpg",
        user_id=1,
    )

    assert result.success is False
    assert result.error == "IMAGE_DOWNLOAD_FAILED"


@pytest.mark.asyncio
async def test_preprocess_upload_failure(
    service: ClosetService, mock_s3_client: MagicMock
) -> None:
    """전처리 실패: S3 업로드 실패 시 에러 반환."""
    mock_s3_client.put_image.side_effect = Exception("Upload failed")

    with patch.object(
        service,
        "_request_presigned_url",
        new_callable=AsyncMock,
        return_value={"fileId": 99, "presignedUrl": "https://s3.example.com/put"},
    ):
        result = await service.preprocess(
            target_image_url="https://example.com/image.jpg",
            user_id=1,
        )

    assert result.success is False
    assert "UPLOAD_FAILED" in result.error


# ── analyze 테스트 ──


@pytest.mark.asyncio
async def test_analyze_success(
    service: ClosetService, mock_analyzer: MagicMock
) -> None:
    """분석 성공: VLM이 정상적으로 결과를 반환."""
    result = await service.analyze(
        target_image_url="https://example.com/image.jpg",
    )

    assert result.success is True
    assert result.major.category == "TOP"
    assert result.major.color == ["흰색"]
    assert result.extra.caption == "흰색 면 티셔츠입니다."
    mock_analyzer.analyze_image.assert_called_once()


@pytest.mark.asyncio
async def test_analyze_download_failure(
    service: ClosetService, mock_s3_client: MagicMock
) -> None:
    """분석 실패: 이미지 다운로드 실패 시 기본값(UNKNOWN) 반환."""
    mock_s3_client.get_image.side_effect = Exception("Download failed")

    result = await service.analyze(
        target_image_url="https://example.com/bad.jpg",
    )

    assert result.success is False
    assert result.major.category == "ETC"
    assert result.error == "IMAGE_DOWNLOAD_FAILED"


@pytest.mark.asyncio
async def test_analyze_vlm_failure_fallback(
    service: ClosetService, mock_analyzer: MagicMock
) -> None:
    """분석 실패: VLM 에러 시 기본값(DEFAULT_ANALYSIS) 사용."""
    mock_analyzer.analyze_image.side_effect = Exception("Gemini API error")

    result = await service.analyze(
        target_image_url="https://example.com/image.jpg",
    )

    # VLM이 에러나도 기본값으로 결과가 반환되어야 함
    assert result.success is True  # 다운로드는 성공, 분석만 fallback
    assert result.major.category == "ETC"
