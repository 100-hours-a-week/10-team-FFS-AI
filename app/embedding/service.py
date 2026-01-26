import logging
from functools import lru_cache
from typing import Any

import httpx
from qdrant_client.http import models as qdrant_models

from app.config import get_settings
from app.core.database import get_qdrant_client
from app.embedding.exceptions import ExternalAPIError, VectorDBError
from app.embedding.formatter import EmbeddingTextFormatter, HybridFormatter
from app.embedding.schemas import EmbeddingRequest

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self: "EmbeddingService", formatter: EmbeddingTextFormatter | None = None
    ) -> None:
        self.formatter: EmbeddingTextFormatter = formatter or HybridFormatter()
        self.settings = get_settings()

    async def get_embedding(self: "EmbeddingService", text: str) -> list[float]:
        if not self.settings.upstage_api_key:
            logger.error("UPSTAGE_API_KEY is not set")
            raise ExternalAPIError("Upstage", "API key is not configured")

        url = "https://api.upstage.ai/v1/solar/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.upstage_api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.settings.embedding_model, "input": text}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=10.0
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

    async def upsert(self: "EmbeddingService", request: EmbeddingRequest) -> bool:
        embedding_text = self.formatter.format(request.metadata, request.caption)
        logger.info("Formatted embedding text: %s...", embedding_text[:100])

        vector = await self.get_embedding(embedding_text)

        qdrant = await get_qdrant_client()
        point_id = request.clothes_id

        payload = {
            "userId": request.user_id,
            "clothesId": request.clothes_id,
            "imageUrl": request.image_url,
            "category": request.metadata.category,
            "color": request.metadata.color,
            "material": request.metadata.material,
            "styleTags": request.metadata.style_tags,
            "gender": request.metadata.gender,
            "season": request.metadata.season,
            "formality": request.metadata.formality,
            "fit": request.metadata.fit,
            "caption": request.caption,
            "embeddingText": embedding_text,  # 디버깅/분석용
        }

        try:
            logger.info("Upserting to Qdrant: point_id=%s", point_id)
            await qdrant.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.info("Successfully upserted point %s", point_id)
            return True
        except Exception as err:
            logger.exception("Failed to upsert to Qdrant: %s", err)
            raise VectorDBError("upsert", str(err)) from err

    async def delete(self: "EmbeddingService", clothes_id: int) -> bool:
        qdrant = await get_qdrant_client()

        try:
            await qdrant.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=qdrant_models.PointIdsList(points=[clothes_id]),
            )
            logger.info("Successfully deleted point %s", clothes_id)
            return True
        except Exception as err:
            logger.exception("Failed to delete from Qdrant: %s", err)
            raise VectorDBError("delete", str(err)) from err


# 기본 서비스 인스턴스 (DI 컨테이너 도입 전까지 사용)
# FastAPI가 의존성으로 분석할 때, 파라미터가 있으면 "요청 파라미터"로 오해할 수 있으므로 제거합니다.
@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
