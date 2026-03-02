"""
Segmentation Service - 콜라주 이미지에서 개별 의류 아이템 추출

Pipeline:
1. generate_collage: Gemini로 모델 이미지 → 플랫레이 콜라주 생성 (여러 장의 이미지 반환)
"""

import logging

from app.closet.gemini_client import GeminiImageAnalyzer

logger = logging.getLogger(__name__)


class SegmentationService:
    """의류 이미지 분리(Segmentation) 서비스"""

    def __init__(self, gemini_client: GeminiImageAnalyzer | None = None) -> None:
        self.gemini_client = gemini_client or GeminiImageAnalyzer()

    async def generate_collage(self, image_url: str) -> list[bytes]:
        """
        모델 착용 이미지에서 플랫레이/개별 아이템 이미지 생성

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            생성된 이미지 bytes 리스트
        """
        logger.info(f"Generating images from: {image_url}")
        images = await self.gemini_client.generate_collage(image_url)
        return images

    async def segment(self, image_url: str) -> list[bytes]:
        """
        전체 Segmentation 파이프라인 실행

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            분리된 개별 아이템 이미지 bytes 리스트
        """
        # 1. 이미지 생성 (Gemini가 여러 장을 생성할 수 있음)
        generated_images = await self.generate_collage(image_url)

        logger.info(
            f"Segmentation complete: {len(generated_images)} items returned from Gemini"
        )
        return generated_images
