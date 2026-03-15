"""커머스 상품 Qdrant 저장소.

commerce 컬렉션에 상품을 upsert하고, 기존 상품 존재 여부를 확인한다.
"""

import logging
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qdrant_models

from app.config import Settings, get_settings
from app.core.database import get_qdrant_client

logger = logging.getLogger(__name__)


def _product_id_to_point_id(product_id: str) -> int:
    """네이버 productId를 Qdrant point_id(int)로 변환."""
    if product_id.isdigit():
        return int(product_id)
    return hash(product_id) % (2**63)


class CommerceRepository:
    """Qdrant commerce 컬렉션 CRUD."""

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._qdrant_client = qdrant_client
        self.settings = settings or get_settings()

    async def _get_qdrant(self) -> AsyncQdrantClient:
        if self._qdrant_client is None:
            self._qdrant_client = await get_qdrant_client()
        return self._qdrant_client

    async def get_existing_product(self, product_id: str) -> dict[str, Any] | None:
        """Qdrant에서 기존 상품을 조회한다.

        Returns:
            존재하면 payload dict, 없으면 None
        """
        qdrant = await self._get_qdrant()
        point_id = _product_id_to_point_id(product_id)

        try:
            results = await qdrant.retrieve(
                collection_name=self.settings.qdrant_shop_collection_name,
                ids=[point_id],
                with_payload=True,
            )
            if results:
                return results[0].payload
            return None
        except Exception:
            return None

    def is_product_changed(
        self, existing_payload: dict[str, Any], product_price: int, product_title: str
    ) -> bool:
        """기존 상품과 비교하여 변동 여부를 확인한다."""
        return (
            existing_payload.get("price") != product_price
            or existing_payload.get("title") != product_title
        )

    async def update_payload(
        self, product_id: str, payload_updates: dict[str, Any]
    ) -> bool:
        """기존 상품의 payload만 업데이트한다 (벡터 재생성 불필요)."""
        qdrant = await self._get_qdrant()
        point_id = _product_id_to_point_id(product_id)

        try:
            await qdrant.set_payload(
                collection_name=self.settings.qdrant_shop_collection_name,
                payload=payload_updates,
                points=[point_id],
            )
            return True
        except Exception as e:
            logger.error(
                "payload 업데이트 실패: product_id=%s, error=%s", product_id, e
            )
            return False

    async def upsert(
        self, product_id: str, vector: list[float], payload: dict[str, Any]
    ) -> bool:
        """상품을 Qdrant에 저장 (신규 또는 전체 갱신)."""
        qdrant = await self._get_qdrant()
        point_id = _product_id_to_point_id(product_id)

        try:
            await qdrant.upsert(
                collection_name=self.settings.qdrant_shop_collection_name,
                points=[
                    qdrant_models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.debug("Qdrant upsert 성공: product_id=%s", product_id)
            return True
        except Exception as e:
            logger.error("Qdrant upsert 실패: product_id=%s, error=%s", product_id, e)
            return False

    async def upsert_batch(
        self,
        points: list[tuple[str, list[float], dict[str, Any]]],
        batch_size: int = 64,
    ) -> int:
        """여러 상품을 배치로 Qdrant에 저장한다.

        Args:
            points: (product_id, vector, payload) 튜플 리스트
            batch_size: 한 번에 upsert할 포인트 수

        Returns:
            성공적으로 저장된 포인트 수
        """
        qdrant = await self._get_qdrant()
        success_count = 0

        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            qdrant_points = [
                qdrant_models.PointStruct(
                    id=_product_id_to_point_id(pid),
                    vector=vec,
                    payload=pay,
                )
                for pid, vec, pay in batch
            ]

            try:
                await qdrant.upsert(
                    collection_name=self.settings.qdrant_shop_collection_name,
                    points=qdrant_points,
                )
                success_count += len(batch)
            except Exception as e:
                logger.error(
                    "배치 upsert 실패: batch %d~%d, error=%s",
                    i,
                    i + len(batch),
                    e,
                )

        return success_count
