"""
Segmentation Service - 콜라주 이미지에서 개별 의류 아이템 추출

Pipeline:
1. generate_collage: Gemini로 모델 이미지 → 플랫레이 콜라주 생성 (여러 장의 이미지 반환)
2. 1순위 모델 실패 시 2순위 Gemini 모델로 자동 폴백
"""

from __future__ import annotations

import logging

from app.closet.gemini_client import GeminiImageAnalyzer

logger = logging.getLogger(__name__)


class SegmentationService:
    """의류 이미지 분리(Segmentation) 서비스"""

    def __init__(
        self,
        gemini_client: GeminiImageAnalyzer | None = None,
    ) -> None:
        self.gemini_client = gemini_client or GeminiImageAnalyzer()

    async def segment(self, image_url: str) -> list[bytes]:
        """
        전체 Segmentation 파이프라인 실행.
        Gemini 내부에서 1순위 → 2순위 모델 폴백을 처리합니다.

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            분리된 개별 아이템 이미지 bytes 리스트
        """
        logger.info(f"Generating images from: {image_url}")
        generated_images = await self.gemini_client.generate_collage(image_url)
        logger.info(
            f"Segmentation complete: {len(generated_images)} items returned from Gemini"
        )
        return generated_images
