"""vLLM 기반 이미지 분석기 — Qwen2.5-VL-7B

ImageAnalyzer 프로토콜을 구현합니다.
GCP L4에서 실행 중인 vLLM 서버의 OpenAI 호환 API를 호출하여
패션 이미지를 분석하고 구조화된 JSON을 반환합니다.
"""

import base64
import json
import logging
import re
from typing import Any

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── 패션 분석 프롬프트 (Colab 검증 완료) ──
ANALYSIS_PROMPT = """너는 글로벌 패션 매거진의 에디터이자 15년 경력의 베테랑 패션 MD야.
제공된 이미지를 전문가의 시각으로 정밀 분석하되, 반드시 약속된 JSON 구조 내에서만 응답해.

### 작성 지침 (데이터 퀄리티):
1. **전문 어휘 사용**: '단순한 빨강' 대신 '버건디', '실버' 대신 '메탈릭 실버' 등 구체적인 색상명을 사용하고, 소재와 핏에 전문 용어를 적용해.
2. **카테고리 분류**: 반드시 다음 6종 중 하나만 선택해: TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC
3. **디테일 반영**: 이미지에서 보이는 특징적인 디테일(넥라인, 실루엣, 워싱 등)을 style_tags와 caption에 충분히 녹여내.

반드시 아래 JSON 구조를 유지하고, 다른 설명 없이 JSON 데이터만 출력해:
{
  "major": {
    "category": "TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC 중 택1",
    "color": ["구체적 색상명"],
    "material": ["소재"],
    "style_tags": ["전문적인 스타일 명칭 및 트렌드 키워드 (예: 시티보이, 올드머니, 고프코어 등)"]
  },
  "extra": {
    "meta_data": {
        "gender": "남성, 여성, 유니섹스 중 택1",
        "season": ["봄", "여름", "가을", "겨울 중 해당하는 것"],
        "formality": "캐주얼, 세미포멀, 비즈니스캐주얼, 포멀 중 가장 적절한 것",
        "fit": "실루엣 및 기장감 (예: 오버사이즈드 핏, 크롭 핏, 와이드 실루엣 등)",
        "occasion": ["적합한 상황/장소"]
    },
    "caption": "이미지의 시각적 특징(실루엣, 디테일, 분위기 등)을 포함하여 전문가의 관점에서 2문장 내외로 서술"
  }
}"""

# ── 카테고리 정규화 ──
VALID_CATEGORIES = {"TOP", "BOTTOM", "DRESS", "SHOES", "ACCESSORY", "ETC"}
CATEGORY_MAP = {
    "OUTERWEAR": "TOP",
    "OUTER": "TOP",
    "COAT": "TOP",
    "JACKET": "TOP",
    "BAG": "ACCESSORY",
    "HAT": "ACCESSORY",
    "SCARF": "ACCESSORY",
    "BELT": "ACCESSORY",
    "WATCH": "ACCESSORY",
    "JEWELRY": "ACCESSORY",
    "GLASSES": "ACCESSORY",
    "SUNGLASSES": "ACCESSORY",
    "SOCKS": "ACCESSORY",
    "SNEAKERS": "SHOES",
    "BOOTS": "SHOES",
    "SANDALS": "SHOES",
    "HEELS": "SHOES",
    "SKIRT": "BOTTOM",
    "PANTS": "BOTTOM",
    "JEANS": "BOTTOM",
    "SHORTS": "BOTTOM",
    "SHIRT": "TOP",
    "BLOUSE": "TOP",
    "SWEATER": "TOP",
    "HOODIE": "TOP",
    "T-SHIRT": "TOP",
    "TSHIRT": "TOP",
    "SUIT": "TOP",
    "JUMPER": "TOP",
    "UNKNOWN": "ETC",
}


class ModelServerAnalyzer:
    """vLLM OpenAI 호환 API를 호출하여 패션 이미지를 분석합니다.

    GCP L4에서 실행 중인 vLLM 서버(Qwen2.5-VL-7B)에 이미지를 전송하고
    분석 결과를 파싱·정규화하여 반환합니다.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self._base_url = base_url or settings.ai_server_url
        self._timeout = timeout
        self._model = "Qwen/Qwen2.5-VL-7B-Instruct"
        logger.info(f"ModelServerAnalyzer 초기화: {self._base_url}")

    async def analyze_image(self, image_bytes: bytes) -> dict[str, Any]:
        """이미지 바이트를 vLLM 서버로 전송하여 분석 결과를 반환합니다.

        Args:
            image_bytes: 분석할 이미지의 바이트 데이터

        Returns:
            정규화된 분석 결과 dict (major/extra 구조)
        """
        # 1. 이미지 → base64 인코딩
        image_b64 = base64.b64encode(image_bytes).decode()

        # 2. OpenAI 호환 요청 구성
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                            },
                        },
                        {
                            "type": "text",
                            "text": ANALYSIS_PROMPT,
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.1,
        }

        # 3. vLLM API 호출
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                )
                response.raise_for_status()

        except httpx.TimeoutException as e:
            logger.error(f"vLLM 서버 타임아웃 ({self._timeout}s)")
            raise RuntimeError("vLLM 서버 응답 타임아웃") from e
        except httpx.ConnectError as e:
            logger.error(f"vLLM 서버 연결 실패: {self._base_url}")
            raise RuntimeError(f"vLLM 서버 연결 불가: {self._base_url}") from e
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM HTTP {e.response.status_code}: {e.response.text[:300]}")
            raise RuntimeError(f"vLLM HTTP {e.response.status_code}") from e

        # 4. 응답 텍스트 추출
        raw_text = response.json()["choices"][0]["message"]["content"]

        # 5. JSON 파싱 + 정규화
        parsed = self._parse_json(raw_text)
        return self._normalize(parsed)

    # ────────────────────────────────
    # 내부 유틸
    # ────────────────────────────────

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """모델 출력 텍스트에서 JSON을 추출합니다."""

        # 1) ```json ... ``` 코드 블록
        match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # 2) { ... } 블록
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        # 3) 실패 → 기본값
        logger.warning(f"JSON 파싱 실패, 기본값 반환. 원본: {text[:200]}")
        return {
            "major": {
                "category": "ETC",
                "color": [],
                "material": [],
                "style_tags": [],
            },
            "extra": {
                "meta_data": {},
                "caption": text[:200] if text else "의류 아이템",
            },
        }

    @staticmethod
    def _normalize_category(raw: str) -> str:
        """카테고리를 허용 6종으로 매핑합니다."""
        if not raw:
            return "ETC"
        upper = raw.strip().upper()
        if upper in VALID_CATEGORIES:
            return upper
        if upper in CATEGORY_MAP:
            logger.info(f"카테고리 매핑: '{raw}' → '{CATEGORY_MAP[upper]}'")
            return CATEGORY_MAP[upper]
        logger.warning(f"알 수 없는 카테고리: '{raw}' → 'ETC'")
        return "ETC"

    @staticmethod
    def _to_list(value: object) -> list:
        """문자열이면 리스트로, None이면 빈 리스트 반환."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return value
        return [str(value)]

    @classmethod
    def _normalize(cls, data: dict) -> dict[str, Any]:
        """모델 출력을 백엔드 스키마에 맞게 정규화합니다."""
        major = data.get("major", {})
        extra = data.get("extra", {})
        meta = extra.get("meta_data", {})

        return {
            "major": {
                "category": cls._normalize_category(major.get("category", "")),
                "color": cls._to_list(major.get("color")),
                "material": cls._to_list(major.get("material")),
                "style_tags": cls._to_list(major.get("style_tags")),
            },
            "extra": {
                "meta_data": {
                    "gender": meta.get("gender"),
                    "season": cls._to_list(meta.get("season")),
                    "formality": meta.get("formality"),
                    "fit": meta.get("fit"),
                    "occasion": cls._to_list(meta.get("occasion")),
                },
                "caption": extra.get("caption") or "의류 아이템",
            },
        }
