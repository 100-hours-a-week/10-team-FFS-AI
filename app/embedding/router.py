import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.embedding.exceptions import ExternalAPIError, VectorDBError
from app.embedding.schemas import DeleteResponse, EmbeddingRequest, EmbeddingResponse
from app.embedding.service import EmbeddingService, get_embedding_service

router = APIRouter(prefix="/v1/closet", tags=["embedding"])
logger = logging.getLogger(__name__)


@router.post(
    "/embedding", response_model=EmbeddingResponse, status_code=status.HTTP_200_OK
)
async def create_embedding(
    request: EmbeddingRequest,
    service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> EmbeddingResponse:
    logger.info(
        "Received embedding request for clothesId: %s, userId: %s",
        request.clothes_id,
        request.user_id,
    )

    try:
        indexed = await service.upsert(request)
        return EmbeddingResponse(
            clothes_id=request.clothes_id,
            caption=request.caption,
            indexed=indexed,
        )
    except ExternalAPIError as err:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"External service error ({err.service}): {err.message}",
        ) from err
    except VectorDBError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database error: {err.message}",
        ) from err


@router.delete(
    "/{clothes_id}", response_model=DeleteResponse, status_code=status.HTTP_200_OK
)
async def delete_embedding(
    clothes_id: int,
    service: Annotated[EmbeddingService, Depends(get_embedding_service)],
) -> DeleteResponse:
    logger.info("Received delete request for clothesId: %s", clothes_id)

    try:
        deleted = await service.delete(clothes_id)
        return DeleteResponse(clothes_id=clothes_id, deleted=deleted)
    except VectorDBError as err:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector database error: {err.message}",
        ) from err
