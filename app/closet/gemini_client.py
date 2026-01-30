import json
import logging
from typing import Any

import google.generativeai as genai
from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory

from app.config import get_settings

logger = logging.getLogger(__name__)


ANALYSIS_PROMPT = """
이 옷의 이미지를 분석해서 다음 정보를 JSON 형식으로 추출해줘:
1. category: 카테고리 (예: 셔츠, 원피스, 바지, 스커트, 자켓, 코트, 패딩 등)
2. color: 색상 목록 (예: ["검정", "흰색"])
3. material: 소재 목록 (예: ["면", "데님", "가죽"])
4. style_tags: 스타일 태그 목록 (예: ["캐주얼", "오버핏", "빈티지"])
5. gender: 성별 (남성, 여성, 유니섹스 중 하나)
6. season: 착용 계절 목록 (예: ["봄", "가을"])
7. formality: 격식 수준 (캐주얼, 세미포멀, 포멀)
8. fit: 핏 (슬림핏, 레귤러핏, 오버핏 등)
9. occasion: 적절한 상황/장소 목록 (예: ["데이트", "출근", "파티"])

추가로 이미지에 대한 자연스러운 설명을 caption 필드에 작성해줘.

JSON 응답 형식:
{
  "major": {
    "category": "...",
    "color": ["..."],
    "material": ["..."],
    "style_tags": ["..."]
  },
  "extra": {
    "meta_data": {
        "gender": "...",
        "season": ["..."],
        "formality": "...",
        "fit": "...",
        "occasion": ["..."]
    },
    "caption": "..."
  }
}
"""


class GeminiImageAnalyzer:
    def __init__(self) -> None:
        self.settings = get_settings()
        if self.settings.gemini_api_key:
            genai.configure(api_key=self.settings.gemini_api_key)
        
        self.model = genai.GenerativeModel(
            model_name=self.settings.gemini_model or "gemini-2.5-flash"
        )

    async def analyze_image(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            safety_settings = [
                {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
                {"category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, "threshold": HarmBlockThreshold.BLOCK_NONE},
            ]

            parts = [
                {"mime_type": "image/jpeg", "data": image_bytes},
                ANALYSIS_PROMPT,
            ]

            generation_config = GenerationConfig(
                response_mime_type="application/json"
            )

            response = await self.model.generate_content_async(
                parts,
                safety_settings=safety_settings,
                generation_config=GenerationConfig(
                    response_mime_type="application/json",
                ),
            )
            
            return self._parse_response(response.text)

        except Exception as e:
            logger.error(f"Gemini analysis failed: {e}")
            raise

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {text}")
            # 파싱 실패시 재시도 로직 추가
            raise ValueError("Invalid JSON response from Gemini") from e
