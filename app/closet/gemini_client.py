import base64
import logging

import httpx
from google import genai
from google.genai import types

from app.common.llm_schemas import ImageAnalysisResult
from app.config import get_settings

logger = logging.getLogger(__name__)

SEGMENTATION_PROMPT = """
Identify all fashion items in the image (dress, shoes, bag, sunglasses, etc.).
You must call the image generation function MULTIPLE TIMES - once for each item found.
For each individual item, generate a separate product photo with:
Only that single item segmented and isolated
Human body parts and background completely removed
Light gray background (#E5E5E5)
Professional studio product shot style
Sharp edges and soft drop shadow
IMPORTANT: Present ALL generated images in your final response. Do not hide them in your thinking process.
"""

ANALYSIS_PROMPT = """
이 옷의 이미지를 분석해서 다음 정보를 JSON 형식으로 추출해줘:
1. category: 카테고리 (반드시 다음 중 하나 선택: TOP, BOTTOM, OUTER, DRESS, SHOES, ACCESSORY, ETC)
2. sub_category: 세부 카테고리 (category에 맞는 값 하나만 선택, ETC는 null)
   - TOP: 반소매_티셔츠, 긴소매_티셔츠, 셔츠_블라우스, 맨투맨_스웨트, 니트_스웨터
   - BOTTOM: 데님_팬츠, 슬랙스_트라우저, 트레이닝_조거, 숏츠, 스커트, 레깅스
   - OUTER: 가디건_니트아우터, 집업_후드아우터, 자켓, 코트, 패딩, 블레이저_수트자켓
   - SHOES: 스니커즈, 로퍼_단화, 구두_힐, 부츠, 샌들_슬리퍼
   - DRESS: 원피스, 점프슈트
   - ACCESSORY: 모자, 스카프_넥, 주얼리, 벨트, 양말_레그웨어, 백팩, 크로스_숄더백, 클러치_파우치, 웨이스트백
3. color: 색상 목록 (예: ["검정", "흰색"])
4. material: 소재 목록 (예: ["면", "데님", "가죽"])
5. style_tags: 스타일 태그 목록 (예: ["캐주얼", "오버핏", "빈티지"])
6. gender: 성별 (남성, 여성, 유니섹스 중 하나)
7. season: 착용 계절 목록 (예: ["봄", "가을"])
8. formality: 격식 수준 (캐주얼, 세미포멀, 포멀)
9. fit: 핏 (슬림핏, 레귤러핏, 오버핏 등)
10. occasion: 적절한 상황/장소 목록 (예: ["데이트", "출근", "파티"])

추가로 이미지에 대한 자연스러운 설명을 caption 필드에 작성해줘.
"""

SAFETY_SETTINGS = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
    types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"
    ),
]


class GeminiImageAnalyzer:
    def __init__(self) -> None:
        self.settings = get_settings()

        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.gemini_model or "gemini-2.5-flash"

    async def analyze_image(self, image_bytes: bytes) -> ImageAnalysisResult:
        try:
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            config = types.GenerateContentConfig(
                safety_settings=SAFETY_SETTINGS,
                response_mime_type="application/json",
                response_schema=ImageAnalysisResult,
            )

            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[ANALYSIS_PROMPT, image_part],
                config=config,
            )

            if resp.parsed is not None:
                return resp.parsed

            text = getattr(resp, "text", None)
            if not text:
                logger.error("Empty response from Gemini, returning fallback")
                return self._fallback()

            return ImageAnalysisResult.model_validate_json(text)

        except Exception:
            logger.exception("Gemini analysis failed")
            raise

    @staticmethod
    def _fallback() -> ImageAnalysisResult:
        from app.common.llm_schemas import (
            ImageExtraAttributes,
            ImageExtraMetadata,
            ImageMajorAttributes,
        )

        return ImageAnalysisResult(
            major=ImageMajorAttributes(category="ETC"),
            extra=ImageExtraAttributes(
                meta_data=ImageExtraMetadata(),
                caption="의류 아이템",
            ),
        )

    async def generate_collage(self, image_url: str) -> list[bytes]:
        """모델 착용 이미지에서 개별 아이템 이미지 생성 (래퍼)."""
        return await self.generate_images(image_url)

    async def generate_images(self, image_url: str) -> list[bytes]:
        """모델 착용 이미지에서 플랫레이/개별 아이템 이미지 생성.

        1순위: vton_model (gemini-3-pro-image-preview)
        2순위: vton_fallback_model (gemini-2.5-flash-image)

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            생성된 이미지 bytes 리스트
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image_bytes = resp.content

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # 1순위: 메인 모델
        primary = self.settings.vton_model
        try:
            return await self._call_gemini_api(primary, b64_image)
        except Exception as primary_err:
            fallback = self.settings.vton_fallback_model
            if not fallback or fallback == primary:
                raise
            logger.warning(
                f"Gemini 1순위 모델({primary}) 실패, "
                f"2순위 모델({fallback})로 재시도: {primary_err}"
            )

        # 2순위: 폴백 모델
        return await self._call_gemini_api(fallback, b64_image)

    async def _call_gemini_api(self, model_id: str, b64_image: str) -> list[bytes]:
        """Gemini API를 호출하여 이미지를 생성합니다."""
        api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models"
            f"/{model_id}:generateContent"
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": SEGMENTATION_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["IMAGE", "TEXT"],
                "temperature": 0.8,
            },
        }

        headers = {"x-goog-api-key": self.settings.gemini_api_key}

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(api_url, json=payload, headers=headers)

            if response.status_code == 200:
                result = response.json()
                generated_images = []

                if "candidates" in result:
                    for candidate in result["candidates"]:
                        for part in candidate.get("content", {}).get("parts", []):
                            if "inlineData" in part:
                                collage_bytes = base64.b64decode(
                                    part["inlineData"]["data"]
                                )
                                logger.info(
                                    f"Collage/Item generated ({model_id}): "
                                    f"{len(collage_bytes)} bytes"
                                )
                                generated_images.append(collage_bytes)

                if generated_images:
                    return generated_images

                raise ValueError(f"No image in Gemini response ({model_id})")

            else:
                error_msg = response.text[:500]
                logger.error(
                    f"Collage generation failed ({model_id}, "
                    f"{response.status_code}): {error_msg}"
                )
                raise ValueError(
                    f"Gemini API error: {response.status_code} ({model_id})"
                )
