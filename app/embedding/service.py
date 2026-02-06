import logging
from functools import lru_cache

from app.closet.schemas import EmbeddingRequest
from app.embedding.client import EmbeddingClient, UpstageEmbeddingClient
from app.embedding.formatter import EmbeddingTextFormatter, HybridFormatter
from app.embedding.repository import QdrantVectorRepository, VectorRepository

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(
        self,
        client: EmbeddingClient | None = None,
        repository: VectorRepository | None = None,
        formatter: EmbeddingTextFormatter | None = None,
    ) -> None:
        self.client = client or UpstageEmbeddingClient()
        self.repository = repository or QdrantVectorRepository()
        self.formatter = formatter or HybridFormatter()

    async def upsert(self, request: EmbeddingRequest) -> bool:
        embedding_text = self.formatter.format(request.major, request.extra)
        logger.info("Formatted embedding text: %s...", embedding_text[:100])

        vector = await self.client.embed(embedding_text)

        payload = self._build_payload(request, embedding_text)

        return await self.repository.upsert(
            point_id=request.clothes_id,
            vector=vector,
            payload=payload,
        )

    async def delete(self, clothes_id: int) -> bool:
        return await self.repository.delete(point_id=clothes_id)

    async def get_embedding(self, text: str) -> list[float]:
        return await self.client.embed(text)

    @staticmethod
    def _build_payload(request: EmbeddingRequest, embedding_text: str) -> dict:
        return {
            "userId": request.user_id,
            "clothesId": request.clothes_id,
            "imageUrl": request.image_url,
            "category": request.major.category,
            "color": request.major.color,
            "material": request.major.material,
            "styleTags": request.major.style_tags,
            "gender": request.extra.meta_data.gender,
            "season": request.extra.meta_data.season,
            "formality": request.extra.meta_data.formality,
            "fit": request.extra.meta_data.fit,
            "occasion": request.extra.meta_data.occasion,
            "caption": request.extra.caption,
            "embeddingText": embedding_text,
        }


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
