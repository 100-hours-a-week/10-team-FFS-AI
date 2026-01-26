import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.embedding.exceptions import ExternalAPIError, VectorDBError
from app.embedding.schemas import EmbeddingRequest, EmbeddingResponse, DeleteResponse
from app.embedding.service import EmbeddingService, get_embedding_service

router = APIRouter(prefix="/v1/closet", tags=["embedding"])
logger = logging.getLogger(__name__)


@router.post("/embedding", response_model=EmbeddingResponse, status_code=status.HTTP_200_OK)
async def create_embedding(
        request: EmbeddingRequest,
        service: Annotated[EmbeddingService, Depends(get_embedding_service)]
):
    """
    이미지 분석 단계에서 생성된 캡션과 메타데이터를 받아 임베딩을 생성하고 벡터 DB(Qdrant)에 저장합니다.
    """
    logger.info(f"Received embedding request for clothesId: {request.clothes_id}, userId: {request.user_id}")

    try:
        indexed = await service.upsert(request)
        return EmbeddingResponse(
            clothes_id=request.clothes_id,
            caption=request.caption,
            indexed=indexed
        )
    except ExternalAPIError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"External service error ({e.service}): {e.message}"
        )
    except VectorDBError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database error: {e.message}"
        )


@router.delete("/{clothesId}", response_model=DeleteResponse, status_code=status.HTTP_200_OK)
async def delete_embedding(
        clothes_id: int,
        service: Annotated[EmbeddingService, Depends(get_embedding_service)]
):
    """
    지정된 clothesId에 해당하는 임베딩을 벡터 DB에서 삭제합니다.
    """
    logger.info(f"Received delete request for clothesId: {clothes_id}")

    try:
        deleted = await service.delete(clothes_id)
        return DeleteResponse(
            clothes_id=clothes_id,
            deleted=deleted
        )
    except VectorDBError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database error: {e.message}"
        )
