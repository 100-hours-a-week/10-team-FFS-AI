from __future__ import annotations

import asyncio
import io
import json
import logging

import httpx
import numpy as np
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

NSFW_THRESHOLD = 0.5

FASHION_PROMPTS = [
    "a product photo of clothing or apparel",
    "a fashion item like a shirt, pants, or dress",
    "a person wearing fashionable clothes",
    "a studio shot of a garment on a plain background",
]

NON_FASHION_PROMPTS = [
    "a photo of an animal or pet",
    "a landscape or nature photography",
    "electronic devices or home appliances",
    "furniture and interior decor objects",
]


class ImageValidator:
    """통합 이미지 검증기 (Ray Serve Client)"""

    def __init__(self: ImageValidator, lazy_load: bool = True) -> None:
        self.server_url = settings.ray_server_url
        if not self.server_url:
            logger.warning("RAY_SERVER_URL is not set. Validation will fail.")

        logger.info(f"ImageValidator Initialized. Target Ray Serve: {self.server_url}")

    async def validate_image(self: ImageValidator, image_url: str) -> dict:
        result = {
            "url": image_url,
            "nsfw": None,
            "fashion": None,
            "embedding": [],
            "error": None,
        }

        try:
            loop = asyncio.get_running_loop()
            image = await loop.run_in_executor(
                None, self._download_image_sync, image_url
            )
        except Exception as e:
            logger.error(f"Download failed for {image_url}: {e}")
            result["error"] = f"DOWNLOAD_FAILED: {str(e)}"
            return result

        if image is None:
            result["error"] = "IMAGE_DOWNLOAD_FAILED"
            return result

        buf = io.BytesIO()
        image.save(buf, format="JPEG")
        image_bytes = buf.getvalue()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # A. NSFW Check (/nsfw)
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

                # B. Fashion Check (/fashion)
                fashion_api = f"{self.server_url}/fashion"
                all_prompts = FASHION_PROMPTS + NON_FASHION_PROMPTS
                files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
                data = {"texts": json.dumps(all_prompts)}

                fashion_resp = await client.post(fashion_api, data=data, files=files)
                fashion_resp.raise_for_status()
                scores = fashion_resp.json()

                scores_np = np.array(scores)
                f_score = float(scores_np[: len(FASHION_PROMPTS)].sum())
                nf_score = float(scores_np[len(FASHION_PROMPTS) :].sum())

                total_score = f_score + nf_score
                confidence = f_score / total_score if total_score > 0 else 0.0
                confidence_threshold = 0.55
                is_fashion = confidence >= confidence_threshold

                debug_scores = {
                    prompt: float(score)
                    for prompt, score in zip(all_prompts, scores, strict=False)
                }
                result["fashion"] = {
                    "is_fashion": is_fashion,
                    "score": confidence,
                    "debug_scores": debug_scores,
                }

        except Exception as e:
            logger.error(f"Ray Serve Validation Error: {e}", exc_info=True)
            result["error"] = str(e)
            result["fashion"] = {"is_fashion": True, "score": 0.0}

        return result

    async def validate_batch(self: ImageValidator, image_urls: list[str]) -> list[dict]:
        tasks = [self.validate_image(url) for url in image_urls]
        results = await asyncio.gather(*tasks)
        return list(results)

    def _download_image_sync(self, url: str) -> Image.Image | None:
        """presigned URL 포함 모든 이미지를 HTTP로 다운로드 (gemini_client와 동일 방식)"""
        try:
            with httpx.Client(timeout=30.0, verify=False) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")
            return None


class MockImageValidator:
    """테스트/로컬 개발용 Mock Validator"""

    async def validate_image(self, url: str) -> dict:
        is_nsfw = "nsfw" in url.lower()
        is_fashion = "food" not in url.lower() and "landscape" not in url.lower()

        return {
            "url": url,
            "nsfw": {"is_nsfw": is_nsfw, "score": 0.9 if is_nsfw else 0.0},
            "fashion": {"is_fashion": is_fashion, "score": 0.99 if is_fashion else 0.1},
            "embedding": [],
            "error": None,
        }

    async def validate_batch(self, urls: list[str]) -> list[dict]:
        tasks = [self.validate_image(u) for u in urls]
        return await asyncio.gather(*tasks)
