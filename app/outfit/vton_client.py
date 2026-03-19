"""
VTON Client - Gemini 3 Pro Image (Nano Banana Pro)
- Google AI Studio API 사용
- 여러 패션 아이템 이미지 → 코디된 모델 이미지 생성
"""

import base64
import logging

import httpx
from langfuse import observe
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class VTONRequest(BaseModel):
    """가상 피팅 요청 스키마"""

    image_urls: list[str] = Field(..., description="패션 아이템 이미지 URL 목록")


class VTONResponse(BaseModel):
    """가상 피팅 응답 스키마"""

    status: str = Field(default="completed", description="상태")
    image_data: bytes | None = Field(default=None, description="생성된 이미지 바이너리")
    error: str | None = Field(default=None, description="에러 메시지")


class VTONClient:
    """Gemini 3 Pro Image 기반 코디 이미지 생성 클라이언트"""

    # 코디 이미지 생성 프롬프트
    OUTFIT_PROMPT = """
[Outfit Coordination Request]
I am providing multiple fashion item images.
Generate a single image of a white mannequin wearing ALL of these items as a coordinated outfit.

[Rules]
1. Use a clean, featureless white mannequin (no face, no skin texture)
2. The mannequin must wear ALL provided items seamlessly
3. Keep each item's original design, color, and texture unchanged
4. Use a clean white or light gray studio background
5. Full body shot, photorealistic, 4K quality, studio lighting
"""

    def __init__(self) -> None:
        from app.config import get_settings

        settings = get_settings()

        self.api_key = settings.gemini_api_key
        self.model_id = settings.vton_model
        self.fallback_model_id = settings.vton_fallback_model

        if self.api_key:
            logger.info(f"VTONClient initialized with primary model: {self.model_id}")
        else:
            logger.error("GOOGLE_API_KEY not found!")

    async def _download_image(self, url: str) -> bytes | None:
        """URL에서 이미지 다운로드"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.content
        except Exception as e:
            logger.warning(f"Failed to download image from {url}: {e}")
            return None

    async def _load_images(self, image_urls: list[str]) -> list[dict]:
        """이미지 URL에서 base64 인코딩된 이미지 목록 생성"""
        image_parts = []

        for i, url in enumerate(image_urls):
            try:
                image_data = await self._download_image(url)
                if image_data:
                    b64 = base64.b64encode(image_data).decode("utf-8")
                    image_parts.append(
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
                    )
                    logger.debug(
                        f"Loaded image {i + 1}: {len(image_data) / 1024:.1f} KB"
                    )
            except Exception as e:
                logger.warning(f"Failed to load image {i + 1}: {e}")
                continue

        return image_parts

    async def _invoke_api(self, payload: dict, model_id: str) -> VTONResponse:
        """단일 모델에 대해 API 호출을 수행합니다."""
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
        headers = {"x-goog-api-key": self.api_key}

        try:
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(api_url, json=payload, headers=headers)

                if response.status_code == 200:
                    result = response.json()
                    if "candidates" in result:
                        for candidate in result["candidates"]:
                            for part in candidate.get("content", {}).get("parts", []):
                                if "inlineData" in part:
                                    image_data = base64.b64decode(
                                        part["inlineData"]["data"]
                                    )
                                    return VTONResponse(
                                        status="completed", image_data=image_data
                                    )
                    return VTONResponse(status="failed", error="No image in response")
                else:
                    error_msg = response.text[:500]
                    return VTONResponse(
                        status="failed",
                        error=f"API Error ({response.status_code}): {error_msg}",
                    )
        except Exception as e:
            return VTONResponse(status="failed", error=str(e))

    @observe(as_type="generation", name="vton_generate_image")
    async def generate_outfit_image(self, request: VTONRequest) -> VTONResponse:
        """
        여러 패션 아이템 이미지로 코디된 모델 이미지 생성 (Fallback 적용)

        Args:
            request: VTONRequest (이미지 URL 목록)

        Returns:
            VTONResponse: 생성된 이미지 또는 에러
        """
        if not self.api_key:
            return VTONResponse(status="failed", error="GOOGLE_API_KEY not configured")

        if not request.image_urls:
            return VTONResponse(status="failed", error="Image URLs are required")

        logger.info(f"Generating outfit image from {len(request.image_urls)} items...")

        # 1. 이미지 로드
        image_parts = await self._load_images(request.image_urls)

        if not image_parts:
            return VTONResponse(status="failed", error="Failed to load any images")

        # 2. API 요청 구성
        parts = [{"text": self.OUTFIT_PROMPT}]
        parts.extend(image_parts)

        payload = {
            "contents": [{"parts": parts}],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "temperature": 0.1,
            },
        }

        # 3. 1순위 모델 호출
        response = await self._invoke_api(payload, self.model_id)
        if response.status == "completed":
            logger.info(
                f"Outfit image generated successfully using primary model: {self.model_id}"
            )
            return response

        # 4. 2순위 (Fallback) 모델 우회 시도
        logger.warning(
            f"메인 모델({self.model_id}) 실패: {response.error}. Fallback 우회 시도..."
        )

        if not self.fallback_model_id or self.fallback_model_id == self.model_id:
            logger.error(
                "Fallback 모델이 설정되지 않았거나 메인 모델과 동일하여 재시도를 포기합니다."
            )
            return response

        fallback_response = await self._invoke_api(payload, self.fallback_model_id)

        if fallback_response.status == "completed":
            logger.info(
                f"Outfit image generated successfully using fallback model: {self.fallback_model_id}"
            )
        else:
            logger.error(
                f"Fallback 모델({self.fallback_model_id})도 실패: {fallback_response.error}"
            )

        return fallback_response
