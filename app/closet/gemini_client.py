import base64
import json
import logging
from typing import Any

import httpx
from google import genai

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
            from google.genai import types

            safety_settings = [
                types.SafetySetting(
                    category="HARM_CATEGORY_HARASSMENT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_HATE_SPEECH",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    threshold="BLOCK_NONE",
                ),
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]

            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

            config = types.GenerateContentConfig(
                safety_settings=safety_settings,
                response_mime_type="application/json",
            )

            resp = await self.client.aio.models.generate_content(
                model=self.model,
                contents=[ANALYSIS_PROMPT, image_part],
                config=config,
            )

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

    async def generate_collage(self, image_url: str) -> bytes:
        """
        모델 착용 이미지에서 플랫레이 콜라주 이미지 생성 (VTON과 동일한 REST API 방식)

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            생성된 이미지 bytes 리스트 (List[bytes])
        """
        # VTON과 동일한 모델 사용
        return await self.generate_images(image_url)

    async def generate_images(self, image_url: str) -> list[bytes]:
        """
        모델 착용 이미지에서 플랫레이/개별 아이템 이미지 생성

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            생성된 이미지 bytes 리스트
        """
        # VTON과 동일한 모델 사용
        model_id = self.settings.vton_model
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"

        # 1. 이미지 다운로드
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            image_bytes = resp.content

        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        # 2. API 요청 구성
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

        # 3. API 호출 (VTON과 동일한 방식)
        headers = {"x-goog-api-key": self.settings.gemini_api_key}

        async with httpx.AsyncClient(timeout=180.0) as client:
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
                                    f"Collage/Item generated: {len(collage_bytes)} bytes"
                                )
                                generated_images.append(collage_bytes)

                if generated_images:
                    return generated_images

                raise ValueError("No image in Gemini response")

            else:
                error_msg = response.text[:500]
                logger.error(
                    f"Collage generation failed ({response.status_code}): {error_msg}"
                )
                raise ValueError(f"Gemini API error: {response.status_code}")
