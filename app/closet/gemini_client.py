import logging

from google import genai
from google.genai import types

from app.common.llm_schemas import ImageAnalysisResult
from app.config import get_settings

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT = """
이 옷의 이미지를 분석해서 다음 정보를 JSON 형식으로 추출해줘:
1. category: 카테고리 (반드시 다음 중 하나 선택: TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC)
2. color: 색상 목록 (예: ["검정", "흰색"])
3. material: 소재 목록 (예: ["면", "데님", "가죽"])
4. style_tags: 스타일 태그 목록 (예: ["캐주얼", "오버핏", "빈티지"])
5. gender: 성별 (남성, 여성, 유니섹스 중 하나)
6. season: 착용 계절 목록 (예: ["봄", "가을"])
7. formality: 격식 수준 (캐주얼, 세미포멀, 포멀)
8. fit: 핏 (슬림핏, 레귤러핏, 오버핏 등)
9. occasion: 적절한 상황/장소 목록 (예: ["데이트", "출근", "파티"])

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
