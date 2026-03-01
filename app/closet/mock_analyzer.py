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
