import json
import logging
from typing import Any

from google import genai

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

        if not self.settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")

        self.client = genai.Client(api_key=self.settings.gemini_api_key)
        self.model = self.settings.gemini_model or "gemini-2.5-flash"

    async def analyze_image(self, image_bytes: bytes) -> dict[str, Any]:
        try:
            # google-genai safety settings 포맷 (카테고리 문자열은 SDK가 허용하는 값 사용)
            # NOTE: 실제 지원 카테고리/표현은 모델/SDK 버전에 따라 다를 수 있어,
            #       여기서는 가장 보편적인 형태로 둠.
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            # image part (bytes)
            image_part = {"mime_type": "image/jpeg", "data": image_bytes}

            # aio 비동기 호출
            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[ANALYSIS_PROMPT, image_part],
                safety_settings=safety_settings,
                config={
                    "response_mime_type": "application/json",
                },
            )

            # SDK 응답은 보통 resp.text 로 바로 접근 가능
            text = getattr(resp, "text", None)
            if not text:
                logger.error("Empty response text from Gemini")
                return self._fallback_parse("")

            return self._parse_response(text)

        except Exception:
            logger.exception("Gemini analysis failed")
            raise

    @staticmethod
    def _parse_response(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse JSON response: %s", text[:500])
            raise ValueError("Invalid JSON response from Gemini") from e

    @staticmethod
    def _fallback_parse(text: str) -> dict[str, Any]:
        return {
            "major": {
                "category": "UNKNOWN",
                "color": [],
                "material": [],
                "style_tags": [],
            },
            "extra": {
                "meta_data": {},
                "caption": text[:200] if text else "의류 아이템",
            },
        }
