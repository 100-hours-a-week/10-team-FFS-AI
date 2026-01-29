from __future__ import annotations

import io
import logging
from typing import Optional

import httpx
from PIL import Image

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BackgroundRemover:
    """
    배경 제거 클라이언트 (Ray Serve Client)

    Ray Serve에 배포된 모델('/segmentation')을 HTTP로 호출하여 배경을 제거합니다.
    URL: settings.AI_MODEL_SERVER_URL (예: http://localhost:8000 또는 http://GCP_IP:8000)
    """

    def __init__(self: BackgroundRemover) -> None:
        self.server_url = settings.ai_model_server_url
        if not self.server_url:
            logger.warning(
                "AI_MODEL_SERVER_URL is not set. Background removal will fail."
            )

        logger.info(
            f"BackgroundRemover Initialized. Target Ray Serve: {self.server_url}"
        )

    async def remove_background(
        self: BackgroundRemover, image: Image.Image
    ) -> Image.Image:
        """
        배경 제거 요청 (/segmentation)
        """
        # Convert Image to Bytes
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_bytes = buf.getvalue()

        api_url = f"{self.server_url}/segmentation"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Send bytes directly (as per Ray Deployment implementation)
                response = await client.post(api_url, content=image_bytes)
                response.raise_for_status()

                # Convert response bytes back to Image
                out_buf = io.BytesIO(response.content)
                return Image.open(out_buf).convert("RGBA")

        except Exception as e:
            logger.error(f"Ray Serve Segmentation Error: {e}")
            # Fail Open: Return original image converted to RGBA
            return image.convert("RGBA")


_remover_instance: Optional[BackgroundRemover] = None


def get_background_remover() -> BackgroundRemover:
    global _remover_instance
    if _remover_instance is None:
        _remover_instance = BackgroundRemover()
    return _remover_instance
