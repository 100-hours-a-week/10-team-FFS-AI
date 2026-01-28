"""
이미지 검증 AI 모델 래퍼

RunPod Serverless에서 실행될 AI 검증 로직을 담은 모듈입니다.

Models:
1. NSFW 검증: Falconsai/nsfw_image_detection
2. 패션 분류: LAION CLIP-ViT-B-32-laion2B-s34B-b79K
3. 중복/유사 검증: Marqo/marqo-fashionSigLIP (임베딩 생성)
"""

from __future__ import annotations

import io
import logging

import httpx
from PIL import Image

from app.core.models import FashionClassifier, NSFWValidator

logger = logging.getLogger(__name__)

# ============================================================
# 설정 상수
# ============================================================

NSFW_MODEL_ID = "Falconsai/nsfw_image_detection"
CLIP_MODEL_ID = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FASHION_SIGLIP_MODEL_ID = "Marqo/marqo-fashionSigLIP"

NSFW_THRESHOLD = 0.5  # NSFW 판정 임계값
FASHION_THRESHOLD = 0.3  # 패션 도메인 판정 임계값

# 패션 관련 텍스트 프롬프트 (CLIP용)
FASHION_PROMPTS = [
    "a photo of clothing",
    "a photo of a fashion item",
    "a photo of a shirt",
    "a photo of pants",
    "a photo of a dress",
    "a photo of shoes",
    "a photo of a jacket",
]

NON_FASHION_PROMPTS = [
    "a photo of food",
    "a photo of a landscape",
    "a photo of a person's face",
    "a photo of an animal",
    "a photo of a building",
    "a photo of text or document",
]


# ============================================================
# 이미지 유틸리티
# ============================================================


def download_image(image_url: str, timeout: float = 30.0) -> Image.Image | None:
    """URL에서 이미지 다운로드"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        with httpx.Client(
            timeout=timeout, follow_redirects=True, verify=False, headers=headers
        ) as client:
            response = client.get(image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
            return image.convert("RGB")
    except Exception as e:
        logger.error(f"이미지 다운로드 실패: {image_url}, 에러: {e}")
        return None


# ============================================================
# 통합 검증기 (Business Logic)
# ============================================================

# Load Model Classes from models.py


class ImageValidator:
    """
    통합 이미지 검증기 (Business Logic Layer)
    """

    def __init__(self: ImageValidator, lazy_load: bool = True) -> None:
        self.nsfw_validator = NSFWValidator()
        self.fashion_classifier = FashionClassifier()

        if not lazy_load:
            self.load_all_models()

    def load_all_models(self: ImageValidator) -> None:
        """모든 모델 로드"""
        logger.info("모든 검증 모델 로딩 시작...")
        self.nsfw_validator.load_model()
        self.fashion_classifier.load_model()
        logger.info("모든 검증 모델 로딩 완료")

    def validate_image(self: ImageValidator, image_url: str) -> dict:
        """이미지 종합 검증"""
        result = {
            "url": image_url,
            "nsfw": None,
            "fashion": None,
            "embedding": [],
            "error": None,
        }

        # 1. 다운로드
        image = download_image(image_url)
        if image is None:
            result["error"] = "IMAGE_DOWNLOAD_FAILED"
            return result

        # 2. NSFW 검증
        try:
            nsfw_output = self.nsfw_validator.predict(image)
            # Output Format: [{"label": "nsfw", "score": 0.95}, ...]

            nsfw_score = 0.0
            for item in nsfw_output:
                if item["label"].lower() == "nsfw":
                    nsfw_score = item["score"]

            is_nsfw = nsfw_score >= NSFW_THRESHOLD
            result["nsfw"] = {"is_nsfw": is_nsfw, "score": nsfw_score}

            if is_nsfw:
                return result  # NSFW면 중단

        except Exception as e:
            logger.error(f"NSFW 검증 에러: {e}")
            result["error"] = f"NSFW_CHECK_FAILED: {e}"
            return result

        # 3. 패션 분류
        try:
            all_prompts = FASHION_PROMPTS + NON_FASHION_PROMPTS
            scores = self.fashion_classifier.get_features(image, all_prompts)

            fashion_score = float(scores[: len(FASHION_PROMPTS)].sum())
            non_fashion_score = float(scores[len(FASHION_PROMPTS) :].sum())

            is_fashion = (
                fashion_score > non_fashion_score and fashion_score >= FASHION_THRESHOLD
            )

            result["fashion"] = {"is_fashion": is_fashion, "score": fashion_score}

            if not is_fashion:
                return result  # 패션 아니면 중단

        except Exception as e:
            logger.error(f"패션 분류 에러: {e}")
            # 에러 시 통과 처리
            result["fashion"] = {"is_fashion": True, "score": 0.0, "error": str(e)}

        return result

    def validate_batch(self: ImageValidator, image_urls: list[str]) -> list[dict]:
        results = []
        for url in image_urls:
            results.append(self.validate_image(url))
        return results


# ============================================================
# Mock 검증기 (테스트용)
# ============================================================


class MockImageValidator:
    """
    테스트용 Mock 검증기

    실제 모델 없이 테스트할 때 사용합니다.
    """

    def validate_image(self: MockImageValidator, image_url: str) -> dict:
        """Mock 검증 결과 반환"""
        # 테스트용 규칙
        is_nsfw = "nsfw" in image_url.lower()
        is_fashion = (
            "food" not in image_url.lower() and "landscape" not in image_url.lower()
        )

        return {
            "url": image_url,
            "nsfw": {"is_nsfw": is_nsfw, "score": 0.9 if is_nsfw else 0.1},
            "fashion": {"is_fashion": is_fashion, "score": 0.8 if is_fashion else 0.2},
            "embedding": [0.1] * 768 if is_fashion and not is_nsfw else [],
            "error": None,
        }

    def validate_batch(self: MockImageValidator, image_urls: list[str]) -> list[dict]:
        """Mock 배치 검증"""
        return [self.validate_image(url) for url in image_urls]
