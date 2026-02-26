from typing import Protocol

from app.common.llm_schemas import ImageAnalysisResult


class ImageAnalyzer(Protocol):
    async def analyze_image(self, image_bytes: bytes) -> ImageAnalysisResult: ...
