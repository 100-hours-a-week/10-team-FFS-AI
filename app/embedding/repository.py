import logging
from typing import Any, Protocol

from qdrant_client.http import models as qdrant_models

from app.config import get_settings
from app.core.database import get_qdrant_client
from app.embedding.exceptions import VectorDBError

logger = logging.getLogger(__name__)


class VectorRepository(Protocol):
    async def upsert(
        self, point_id: int, vector: list[float], payload: dict[str, Any]
    ) -> bool: ...

    async def delete(self, point_id: int) -> bool: ...


class QdrantVectorRepository:
    def __init__(self) -> None:
        self.settings = get_settings()

    async def upsert(
        self, point_id: int, vector: list[float], payload: dict[str, Any]
    ) -> bool:
        qdrant = await get_qdrant_client()

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

    async def delete(self, point_id: int) -> bool:
        qdrant = await get_qdrant_client()

        try:
            await qdrant.delete(
                collection_name=self.settings.qdrant_collection_name,
                points_selector=qdrant_models.PointIdsList(points=[point_id]),
            )
            logger.info("Successfully deleted point %s", point_id)
            return True
        except Exception as err:
            logger.exception("Failed to delete from Qdrant: %s", err)
            raise VectorDBError("delete", str(err)) from err
