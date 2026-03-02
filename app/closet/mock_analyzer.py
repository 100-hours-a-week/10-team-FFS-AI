import asyncio
import logging

from app.common.llm_schemas import (
    ImageAnalysisResult,
    ImageExtraAttributes,
    ImageExtraMetadata,
    ImageMajorAttributes,
)

logger = logging.getLogger(__name__)

DEFAULT_MOCK_RESULT = ImageAnalysisResult(
    major=ImageMajorAttributes(
        category="TOP",
        color=["블루", "화이트"],
        material=["면", "폴리에스터"],
        style_tags=["캐주얼", "스트리트"],
    ),
    extra=ImageExtraAttributes(
        meta_data=ImageExtraMetadata(
            gender="유니섹스",
            season=["봄", "여름", "가을"],
            formality="캐주얼",
            fit="레귤러핏",
            occasion=["데이트", "캠퍼스", "외출"],
        ),
        caption="Mock analysis result for load testing",
    ),
)


class MockImageAnalyzer:
    def __init__(self, delay_seconds: float = 4.0) -> None:
        self.delay_seconds = delay_seconds
        logger.info(
            "MockImageAnalyzer initialized (delay: %ss)",
            delay_seconds,
        )

    async def analyze_image(self, image_bytes: bytes) -> ImageAnalysisResult:
        await asyncio.sleep(self.delay_seconds)
        logger.debug("Mock analysis completed (%ss)", self.delay_seconds)
        return DEFAULT_MOCK_RESULT


# ── Mock 세그멘테이션 ──

# 1x1 PNG (최소 유효 PNG 파일, 87 bytes)
_MOCK_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class MockSegmentationService:
    """Mock 세그멘테이션 — 실제 API 호출 없이 고정 3개 아이템을 반환."""

    def __init__(self, item_count: int = 3, delay_seconds: float = 9.0) -> None:
        self.item_count = item_count
        self.delay_seconds = delay_seconds
        logger.info(
            "MockSegmentationService initialized (items: %d, delay: %ss)",
            item_count,
            delay_seconds,
        )

    async def segment(self, image_url: str) -> list[bytes]:
        await asyncio.sleep(self.delay_seconds)
        logger.info(
            "Mock segmentation completed: %d items (%ss)",
            self.item_count,
            self.delay_seconds,
        )
        return [_MOCK_PNG] * self.item_count
