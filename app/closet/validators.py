from __future__ import annotations

import io
import logging

import httpx
import numpy as np
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Constants for prompts (Client sends these to Ray Serve)
NSFW_THRESHOLD = 0.5
FASHION_THRESHOLD = 0.3
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


class ImageValidator:
    """
    통합 이미지 검증기 (Ray Serve Client)

    Ray Serve에 배포된 모델('/nsfw', '/fashion')을 HTTP로 호출하여 검증합니다.
    URL: settings.AI_MODEL_SERVER_URL (예: http://localhost:8000 또는 http://GCP_IP:8000)
    """

    def __init__(self: ImageValidator, lazy_load: bool = True) -> None:
        # RUN_MODE 제거, 오직 Server URL만 의존
        self.server_url = settings.ai_model_server_url
        if not self.server_url:
            logger.warning("AI_MODEL_SERVER_URL is not set. Validation will fail.")

        logger.info(f"ImageValidator Initialized. Target Ray Serve: {self.server_url}")

    async def validate_image(self: ImageValidator, image_url: str) -> dict:
        """
        단일 이미지 검증
        1. NSFW 모델 호출 (/nsfw)
        2. Fashion 모델 호출 (/fashion)
        """
        result = {
            "url": image_url,
            "nsfw": None,
            "fashion": None,
            "embedding": [],
            "error": None,
        }

        # 1. Download for NSFW (Sending Bytes)
        # Note: Optimization possible if NSFW model supports URL download.
        # Currently, we maintain the logic of sending bytes for NSFW.
        image = self._download_image_sync(image_url)
        if image is None:
            result["error"] = "IMAGE_DOWNLOAD_FAILED"
            return result

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # ---------------------------
                # A. NSFW Check (/nsfw)
                # ---------------------------
                nsfw_api = f"{self.server_url}/nsfw"
                nsfw_resp = await client.post(nsfw_api, content=image_bytes)
                nsfw_resp.raise_for_status()
                nsfw_out = nsfw_resp.json()

                nsfw_score = 0.0
                for item in nsfw_out:
                    if item["label"].lower() == "nsfw":
                        nsfw_score = item["score"]
                is_nsfw = nsfw_score >= NSFW_THRESHOLD
                result["nsfw"] = {"is_nsfw": is_nsfw, "score": nsfw_score}

                if is_nsfw:
                    logger.info(f"Image {image_url} blocked by NSFW filter.")
                    return result

                # ---------------------------
                # B. Fashion Check (/fashion)
                # ---------------------------
                fashion_api = f"{self.server_url}/fashion"
                all_prompts = FASHION_PROMPTS + NON_FASHION_PROMPTS

                # Protocol: Send JSON with image_url and prompts
                payload = {"image_url": image_url, "texts": all_prompts}

                fashion_resp = await client.post(fashion_api, json=payload)
                fashion_resp.raise_for_status()
                scores = fashion_resp.json()  # list[float]

                scores_np = np.array(scores)
                f_score = float(scores_np[: len(FASHION_PROMPTS)].sum())
                nf_score = float(scores_np[len(FASHION_PROMPTS) :].sum())
                is_fashion = f_score > nf_score and f_score >= FASHION_THRESHOLD
                result["fashion"] = {"is_fashion": is_fashion, "score": f_score}

        except Exception as e:
            logger.error(f"Ray Serve Validation Error: {e}")
            result["error"] = str(e)
            result["fashion"] = {"is_fashion": True, "score": 0.0}  # Fail open

        return result

    async def validate_batch(self: ImageValidator, image_urls: list[str]) -> list[dict]:
        """
        배치 검증
        단순 반복 호출로 구현 (Ray Serve가 내부적으로 동시성 처리)
        """
        results = []
        for url in image_urls:
            res = await self.validate_image(url)
            results.append(res)
        return results

    def _download_image_sync(self, url: str) -> Image.Image | None:
        try:
            with httpx.Client(timeout=30.0, verify=False) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception:
            return None


class MockImageValidator:
    """테스트용 Mock"""

    def validate_image(self, url: str) -> dict:
        is_nsfw = "nsfw" in url.lower()
        is_fashion = "food" not in url.lower() and "landscape" not in url.lower()

        return {
            "url": url,
            "nsfw": {"is_nsfw": is_nsfw, "score": 0.9 if is_nsfw else 0.0},
            "fashion": {"is_fashion": is_fashion, "score": 0.99 if is_fashion else 0.1},
            "embedding": [],
            "error": None,
        }

    def validate_batch(self, urls: list[str]) -> list[dict]:
        return [self.validate_image(u) for u in urls]


# Compatibility utility for service.py
def download_image(image_url: str, timeout: float = 30.0):
    try:
        with httpx.Client(timeout=timeout, verify=False) as client:
            resp = client.get(image_url)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None
