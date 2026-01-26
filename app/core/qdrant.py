import logging
import os
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

# 환경변수 로드
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "fashion_items")
VECTOR_SIZE = 768  # Marqo-FashionSigLIP 임베딩 크기


class QdrantService:
    """Qdrant 벡터 DB 관리 서비스"""

    def __init__(self) -> None:  # noqa: ANN101
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self.collection_name = COLLECTION_NAME
        self._ensure_collection()

    def _ensure_collection(self) -> None:  # noqa: ANN101
        """컬렉션 존재 여부 확인 및 생성"""
        try:
            collections = self.client.get_collections()
            exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            if not exists:
                logger.info(
                    f"Qdrant 컬렉션 생성: {self.collection_name} (size: {VECTOR_SIZE})"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=VECTOR_SIZE, distance=models.Distance.COSINE
                    ),
                )
            else:
                logger.debug(f"Qdrant 컬렉션 이미 존재: {self.collection_name}")
        except Exception as e:
            logger.error(f"Qdrant 컬렉션 확인/생성 실패: {e}")
            # 로컬 개발 환경에서 컨테이너가 아직 안 떴을 수 있음
            pass

    def search_similar(
        self,  # noqa: ANN101
        vector: list[float],
        limit: int = 1,
        score_threshold: float = 0.0,
        user_id: int | None = None,
    ) -> list[models.ScoredPoint]:
        """
        유사 벡터 검색

        Args:
            vector: 검색할 임베딩 벡터
            limit: 반환할 결과 수
            score_threshold: 최소 유사도 점수 (0.0 ~ 1.0)
            user_id: 특정 사용자 ID로 필터링 (선택)
        """
        try:
            query_filter = None
            if user_id is not None:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        )
                    ]
                )

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            ).points
            return results
        except Exception as e:
            logger.error(f"Qdrant 검색 실패: {e}")
            return []

    def upsert_item(
        self,  # noqa: ANN101
        id: int,
        vector: list[float],
        payload: dict[str, Any],  # noqa: ANN101
    ) -> bool:
        """
        아이템 저장 (Insert or Update)

        Args:
            id: 아이템 ID (clothesId와 동일하게 사용)
            vector: 임베딩 벡터
            payload: 저장할 메타데이터 (user_id, category 등)
        """
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=[models.PointStruct(id=id, vector=vector, payload=payload)],
            )
            logger.info(f"Qdrant 아이템 저장 성공: ID {id}")
            return True
        except Exception as e:
            logger.error(f"Qdrant 저장 실패 (ID {id}): {e}")
            return False

    def delete_item(self, id: int) -> bool:  # noqa: ANN101
        """아이템 삭제"""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=[id]),
            )
            logger.info(f"Qdrant 아이템 삭제 성공: ID {id}")
            return True
        except Exception as e:
            logger.error(f"Qdrant 삭제 실패 (ID {id}): {e}")
            return False


# 싱글톤 인스턴스 (필요 시)
# qdrant_service = QdrantService()
