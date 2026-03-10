"""vLLM 기반 이미지 분석기 — Gemma-3-12B-it-INT8

ImageAnalyzer 프로토콜을 구현합니다.
GCP L4에서 실행 중인 vLLM 서버(Gemma-3 12B)의 OpenAI 호환 API를 호출하여
패션 이미지를 분석하고 구조화된 JSON을 반환합니다.
"""

import base64
import io
import json
import logging
import re
from typing import Any

import httpx
from PIL import Image

from app.common.llm_schemas import (
    CATEGORY_DETAIL,
    ImageAnalysisResult,
    ImageExtraAttributes,
    ImageExtraMetadata,
    ImageMajorAttributes,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

# ── 패션 분석 프롬프트 (Colab 검증 완료) ──
ANALYSIS_PROMPT = """
[가장 중요한 규칙]
모든 JSON 필드의 값(Value)은 반드시 **'한국어(Korean)'**로만 작성해줘 (caption 포함).
단, `category`와 `sub_category` 필드의 값은 반드시 제공된 정해진 값만 사용해야 해.

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
7. season: 착용 계절 목록 (예: ["봄", "여름", "가을", "겨울 중 해당하는 것"])
8. formality: 격식 수준 (캐주얼, 세미포멀, 비즈니스캐주얼, 포멀 중 가장 적절한 것)
9. fit: 핏 (슬림핏, 레귤러핏, 오버핏 등)
10. occasion: 적절한 상황/장소 목록 (예: ["데이트", "출근", "파티"])

추가로 이미지에 대한 자연스러운 설명을 caption 필드에 작성해줘.

JSON 응답 형식:
{
  "major": {
    "category": "...",
    "sub_category": "...",
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

# ── 카테고리 정규화 ──
# OUTER가 VALID_CATEGORIES에 포함되어 직접 통과 처리됨
VALID_CATEGORIES = {"TOP", "BOTTOM", "OUTER", "DRESS", "SHOES", "ACCESSORY", "ETC"}
CATEGORY_MAP: dict[str, str] = {
    # TOP 계열 — 기존 보존
    "SHIRT": "TOP",
    "BLOUSE": "TOP",
    "SWEATER": "TOP",
    "KNIT": "TOP",
    "KNITWEAR": "TOP",
    "SWEATSHIRT": "TOP",
    "HOODIE": "TOP",
    "T-SHIRT": "TOP",
    "TSHIRT": "TOP",
    "TEE": "TOP",
    "PULLOVER": "TOP",
    # OUTER 계열 — 기존 TOP에서 OUTER로 재매핑 + 신규 추가
    "OUTERWEAR": "OUTER",
    "COAT": "OUTER",
    "JACKET": "OUTER",
    "JUMPER": "OUTER",
    "CARDIGAN": "OUTER",
    "BLAZER": "OUTER",
    "SUIT": "OUTER",
    "VEST": "OUTER",
    "PADDING": "OUTER",
    # BOTTOM 계열 — 기존 보존 + 신규 추가
    "PANTS": "BOTTOM",
    "JEANS": "BOTTOM",
    "SKIRT": "BOTTOM",
    "SHORTS": "BOTTOM",
    "LEGGINGS": "BOTTOM",
    "TROUSERS": "BOTTOM",
    # SHOES 계열 — 기존 보존 + 신규 추가
    "SNEAKERS": "SHOES",
    "BOOTS": "SHOES",
    "SANDALS": "SHOES",
    "HEELS": "SHOES",
    "LOAFERS": "SHOES",
    "FLATS": "SHOES",
    # ACCESSORY 계열 — 기존 보존
    "BAG": "ACCESSORY",
    "BACKPACK": "ACCESSORY",
    "HAT": "ACCESSORY",
    "CAP": "ACCESSORY",
    "SCARF": "ACCESSORY",
    "BELT": "ACCESSORY",
    "JEWELRY": "ACCESSORY",
    "SOCKS": "ACCESSORY",
    "WATCH": "ACCESSORY",
    "GLASSES": "ACCESSORY",
    "SUNGLASSES": "ACCESSORY",
    # DRESS 계열 — 신규 추가
    "JUMPSUIT": "DRESS",
    "ONE-PIECE": "DRESS",
    # ETC
    "UNKNOWN": "ETC",
}


class ModelServerAnalyzer:
    """vLLM OpenAI 호환 API를 호출하여 패션 이미지를 분석합니다.

    GCP L4에서 실행 중인 vLLM 서버(Gemma-3 12B)에 이미지를 전송하고
    분석 결과를 파싱·정규화하여 반환합니다.
    """

    def __init__(self, base_url: str | None = None, timeout: float = 120.0) -> None:
        settings = get_settings()
        self._base_url = base_url or settings.vllm_server_url
        self._timeout = timeout
        self._model = settings.vllm_model_name
        logger.info(f"ModelServerAnalyzer 초기화: {self._base_url}")

    async def analyze_image(self, image_bytes: bytes) -> ImageAnalysisResult:
        """이미지 바이트를 vLLM 서버로 전송하여 분석 결과를 반환합니다.

        Args:
            image_bytes: 분석할 이미지의 바이트 데이터

        Returns:
            ImageAnalysisResult: 정규화된 분석 결과 (Pydantic 모델)
        """
        # 1. 이미지 해상도 최적화 (Resizing)
        # 이미지 크기가 너무 크면 vLLM 메모리/연산 시간에 막대한 악영향을 미침 (12B 속도 최적화)
        try:
            img = Image.open(io.BytesIO(image_bytes))

            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")

            max_size = (768, 768)
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

            output_buffer = io.BytesIO()
            img.save(output_buffer, format="JPEG", quality=85)
            image_bytes = output_buffer.getvalue()

            logger.info(
                f"이미지 최적화 성공: {img.size} 해상도 ({len(image_bytes)} bytes)"
            )

        except Exception as e:
            logger.warning(f"이미지 최적화 실패 (원본 이미지로 진행): {e}")

        # 2. 이미지 → base64 인코딩
        image_b64 = base64.b64encode(image_bytes).decode()

        # 3. OpenAI 호환 요청 구성
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
                    f"{self._base_url.rstrip('/')}/v1/chat/completions",
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
        else:
            # 4. 응답 텍스트 추출
            raw_text = response.json()["choices"][0]["message"]["content"]

            # 5. JSON 파싱 + 정규화 → ImageAnalysisResult 반환
            parsed = self._parse_json(raw_text)
            normalized = self._normalize(parsed)
            return self._to_result(normalized)

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

    @staticmethod
    def _to_result(data: dict[str, Any]) -> ImageAnalysisResult:
        """정규화된 dict를 ImageAnalysisResult Pydantic 모델로 변환합니다."""
        major = data["major"]
        extra = data["extra"]
        meta = extra.get("meta_data", {})

        return ImageAnalysisResult(
            major=ImageMajorAttributes(
                category=major["category"],
                sub_category=major.get("sub_category"),
                color=major.get("color", []),
                material=major.get("material", []),
                style_tags=major.get("style_tags", []),
            ),
            extra=ImageExtraAttributes(
                meta_data=ImageExtraMetadata(
                    gender=meta.get("gender"),
                    season=meta.get("season", []),
                    formality=meta.get("formality"),
                    fit=meta.get("fit"),
                    occasion=meta.get("occasion", []),
                ),
                caption=extra.get("caption") or "의류 아이템",
            ),
        )

    @classmethod
    def _normalize(cls, data: dict) -> dict[str, Any]:
        """모델 출력을 백엔드 스키마에 맞게 정규화합니다."""
        major = data.get("major", {})
        extra = data.get("extra", {})
        meta = extra.get("meta_data", {})

        # L1 카테고리 정규화
        normalized_category = cls._normalize_category(major.get("category", ""))

        # L2 sub_category: VLM이 선택한 값이 해당 L1의 허용 목록에 있을 때만 채택
        raw_sub = major.get("sub_category")
        valid_subs = CATEGORY_DETAIL.get(normalized_category, [])
        sub_category = raw_sub if raw_sub in valid_subs else None

        return {
            "major": {
                "category": normalized_category,
                "sub_category": sub_category,
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
