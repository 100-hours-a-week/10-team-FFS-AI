"""
Segmentation Service - 콜라주 이미지에서 개별 의류 아이템 추출

Pipeline:
1. generate_collage: Gemini로 모델 이미지 → 플랫레이 콜라주 생성 (여러 장의 이미지 반환)
2. Gemini 실패 시 OpenAI 폴백
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.closet.gemini_client import GeminiImageAnalyzer

if TYPE_CHECKING:
    from app.closet.openai_client import OpenAISegmentationClient

logger = logging.getLogger(__name__)


class SegmentationService:
    """의류 이미지 분리(Segmentation) 서비스"""

    def __init__(
        self,
        gemini_client: GeminiImageAnalyzer | None = None,
        openai_client: OpenAISegmentationClient | None = None,
    ) -> None:
        self.gemini_client = gemini_client or GeminiImageAnalyzer()
        self._fallback_client = openai_client

    async def segment(self, image_url: str) -> list[bytes]:
        """
        전체 Segmentation 파이프라인 실행.
        Gemini 실패 시 OpenAI로 폴백합니다.

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            분리된 개별 아이템 이미지 bytes 리스트
        """
        try:
            logger.info(f"Generating images from: {image_url}")
            generated_images = await self.gemini_client.generate_collage(image_url)
            logger.info(
                f"Segmentation complete: {len(generated_images)} items returned from Gemini"
            )
            return generated_images
        except Exception as e:
            if self._fallback_client is None:
                raise
            logger.warning(f"Gemini 세그멘테이션 실패, OpenAI 폴백 시도: {e}")
            generated_images = await self._fallback_client.segment(image_url)
            logger.info(
                f"Segmentation complete (OpenAI fallback): "
                f"{len(generated_images)} items"
            )
            return generated_images
