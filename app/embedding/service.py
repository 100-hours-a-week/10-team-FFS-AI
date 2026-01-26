import logging
from typing import List

import httpx
from qdrant_client.http import models as qdrant_models

from app.config import get_settings
from app.core.database import get_qdrant_client
from app.embedding.exceptions import ExternalAPIError, VectorDBError
from app.embedding.formatter import EmbeddingTextFormatter, HybridFormatter
from app.embedding.schemas import EmbeddingRequest

logger = logging.getLogger(__name__)


class EmbeddingService:

    def __init__(self, formatter: EmbeddingTextFormatter | None = None):
        self.formatter = formatter or HybridFormatter()
        self.settings = get_settings()

    async def get_embedding(self, text: str) -> List[float]:
        if not self.settings.upstage_api_key:
            logger.error("UPSTAGE_API_KEY is not set")
            raise ExternalAPIError("Upstage", "API key is not configured")

        url = "https://api.upstage.ai/v1/solar/embeddings"
        headers = {
            "Authorization": f"Bearer {self.settings.upstage_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.settings.embedding_model,
            "input": text
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url, headers=headers, json=payload, timeout=10.0
                )
                response.raise_for_status()
                result = response.json()
                return result["data"][0]["embedding"]
        except httpx.HTTPStatusError as e:
            logger.error(f"Upstage API error: {e.response.text}")
            raise ExternalAPIError("Upstage", e.response.text)
        except httpx.TimeoutException:
            logger.error("Upstage API timeout")
            raise ExternalAPIError("Upstage", "Request timeout")
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise ExternalAPIError("Upstage", str(e))

    async def upsert(self, request: EmbeddingRequest) -> bool:
        embedding_text = self.formatter.format(request.metadata, request.caption)
        logger.info(f"Formatted embedding text: {embedding_text[:100]}...")

        vector = await self.get_embedding(embedding_text)

        qdrant = await get_qdrant_client()
        point_id = request.clothesId

        payload = {
            "userId": request.userId,
            "clothesId": request.clothesId,
            "imageUrl": request.imageUrl,
            "category": request.metadata.category,
            "color": request.metadata.color,
            "material": request.metadata.material,
            "styleTags": request.metadata.styleTags,
            "gender": request.metadata.gender,
            "season": request.metadata.season,
            "formality": request.metadata.formality,
            "fit": request.metadata.fit,
            "caption": request.caption,
            "embeddingText": embedding_text,  # 디버깅/분석용
        }

        try:
            logger.info(f"Upserting to Qdrant: point_id={point_id}")
            await qdrant.upsert(
                collection_name=self.settings.qdrant_collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            logger.info(f"Successfully upserted point {point_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert to Qdrant: {e}")
            raise VectorDBError("upsert", str(e))

    async def delete(self, clothes_id: int) -> bool:
        qdrant = await get_qdrant_client()

        try:
            await qdrant.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=qdrant_models.PointIdsList(
                    points=[clothes_id]
                )
            )
            logger.info(f"Successfully deleted point {clothes_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete from Qdrant: {e}")
            raise VectorDBError("delete", str(e))


# 기본 서비스 인스턴스 (DI 컨테이너 도입 전까지 사용)
def get_embedding_service(
        formatter: EmbeddingTextFormatter | None = None
) -> EmbeddingService:
    return EmbeddingService(formatter=formatter)
