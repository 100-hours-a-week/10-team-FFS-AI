import logging
from typing import Any, Protocol

import httpx

from app.config import get_settings
from app.embedding.exceptions import ExternalAPIError

logger = logging.getLogger(__name__)


class EmbeddingClient(Protocol):

    async def embed(self, text: str) -> list[float]:
        ...

class UpstageEmbeddingClient:

    def __init__(self) -> None:
        self.settings = get_settings()
        self.base_url = "https://api.upstage.ai/v1/solar/embeddings"

    async def embed(self, text: str) -> list[float]:
        if not self.settings.upstage_api_key:
            logger.error("UPSTAGE_API_KEY is not set")
            raise ExternalAPIError("Upstage", "API key is not configured")

        headers = {
            "Authorization": f"Bearer {self.settings.upstage_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.settings.embedding_model, "input": text}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url, headers=headers, json=payload, timeout=10.0
                )
                response.raise_for_status()
                result: dict[str, Any] = response.json()
                return result["data"][0]["embedding"]
        except httpx.HTTPStatusError as err:
            logger.error("Upstage API error: %s", err.response.text)
            raise ExternalAPIError("Upstage", err.response.text) from err
        except httpx.TimeoutException as err:
            logger.error("Upstage API timeout")
            raise ExternalAPIError("Upstage", "Request timeout") from err
        except Exception as err:
            logger.exception("Unexpected error during embedding: %s", err)
            raise ExternalAPIError("Upstage", str(err)) from err