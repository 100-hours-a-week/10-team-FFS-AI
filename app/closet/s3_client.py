import logging

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class S3Client:


    def __init__(self) -> None:
        self.timeout = 30.0

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def get_image(self, url: str) -> bytes:

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url)
            response.raise_for_status()
            return response.content

    @staticmethod
    def _detect_content_type(data: bytes) -> str:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        # 기본값
        return "image/jpeg"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    async def put_image(
        self, presigned_url: str, image_bytes: bytes, content_type: str | None = None
    ) -> None:
        if content_type is None:
            content_type = self._detect_content_type(image_bytes)

        headers = {
            "Content-Type": content_type,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.put(
                presigned_url, content=image_bytes, headers=headers
            )
            response.raise_for_status()
