"""OpenAI 기반 세그멘테이션 클라이언트 — Gemini 폴백용

Gemini 세그멘테이션 실패(503 등) 시 OpenAI를 사용하여
모델 착용 이미지에서 개별 의류 아이템을 분리합니다.

흐름:
1. GPT-4o로 이미지 속 패션 아이템 식별
2. gpt-image-1.5로 아이템별 개별 상품 사진 생성
"""

import base64
import json
import logging

import httpx
import openai

from app.config import get_settings

logger = logging.getLogger(__name__)

IDENTIFY_PROMPT = """
이 이미지에서 착용 중인 패션 아이템을 모두 식별해줘.
각 아이템을 간결한 한국어로 설명 (예: "흰색 반팔 티셔츠", "청바지", "흰색 스니커즈").

반드시 JSON 배열로만 응답해줘. 다른 텍스트 없이 배열만:
["아이템1", "아이템2", "아이템3"]
"""

ISOLATE_PROMPT_TEMPLATE = """
이 이미지에서 '{item}'만 분리해줘.
- 해당 아이템만 남기고 사람 몸, 배경 모두 제거
- 밝은 회색 배경 (#E5E5E5)
- 전문 스튜디오 상품 사진 스타일
- 선명한 경계선과 부드러운 그림자
"""


class OpenAISegmentationClient:
    """OpenAI API를 사용한 세그멘테이션 폴백 클라이언트."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set")

        self._client = openai.AsyncOpenAI(
            api_key=settings.openai_api_key,
            timeout=float(settings.llm_timeout),
            max_retries=settings.llm_max_retries,
        )

    async def segment(self, image_url: str) -> list[bytes]:
        """모델 착용 이미지에서 개별 아이템 이미지를 생성합니다.

        Args:
            image_url: 원본 모델 이미지 URL

        Returns:
            생성된 이미지 bytes 리스트
        """
        # 1. 이미지 다운로드
        image_bytes = await self._download_image(image_url)

        # 2. GPT-4o로 아이템 식별
        items = await self._identify_items(image_bytes)
        if not items:
            raise ValueError("OpenAI: 이미지에서 패션 아이템을 식별하지 못했습니다")

        logger.info(f"OpenAI 아이템 식별 완료: {items}")

        # 3. 아이템별 이미지 생성
        results: list[bytes] = []
        for item in items:
            try:
                img = await self._isolate_item(image_bytes, item)
                if img:
                    logger.info(f"OpenAI 아이템 분리 완료: '{item}' ({len(img)} bytes)")
                    results.append(img)
            except Exception as e:
                logger.warning(f"OpenAI 아이템 분리 실패 ('{item}'): {e}")

        if not results:
            raise ValueError("OpenAI: 아이템 이미지 생성에 모두 실패했습니다")

        return results

    async def _download_image(self, image_url: str) -> bytes:
        """이미지 URL에서 이미지를 다운로드합니다."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(image_url)
            resp.raise_for_status()
            return resp.content

    async def _identify_items(self, image_bytes: bytes) -> list[str]:
        """GPT-4o로 이미지 속 패션 아이템을 식별합니다."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")

        resp = await self._client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IDENTIFY_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                            },
                        },
                    ],
                }
            ],
            temperature=0.1,
            max_tokens=256,
        )

        text = resp.choices[0].message.content or ""

        try:
            items = json.loads(text.strip())
            if isinstance(items, list):
                return [str(item) for item in items]
        except json.JSONDecodeError:
            pass

        # JSON 파싱 실패 시 배열 부분만 추출 시도
        import re

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group())
                return [str(item) for item in items]
            except json.JSONDecodeError:
                pass

        logger.warning(f"아이템 식별 JSON 파싱 실패: {text[:200]}")
        return []

    async def _isolate_item(self, image_bytes: bytes, item: str) -> bytes | None:
        """gpt-image-1.5로 특정 아이템만 분리한 이미지를 생성합니다."""
        prompt = ISOLATE_PROMPT_TEMPLATE.format(item=item)

        result = await self._client.images.edit(
            model="gpt-image-1.5",
            image=image_bytes,
            prompt=prompt,
            size="1024x1024",
        )

        if result.data and result.data[0].b64_json:
            return base64.b64decode(result.data[0].b64_json)

        return None
