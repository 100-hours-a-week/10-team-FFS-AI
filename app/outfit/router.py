import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.schemas import OutfitRequest, OutfitResponse
from app.outfit.service import OutfitService, get_outfit_service

router = APIRouter(prefix="/v1/closet", tags=["outfit"])
logger = logging.getLogger(__name__)


@router.post("/outfit", response_model=OutfitResponse, status_code=status.HTTP_200_OK)
async def recommend_outfit(
    request: OutfitRequest,
    service: Annotated[OutfitService, Depends(get_outfit_service)],
) -> OutfitResponse:
    trace_id = str(uuid.uuid4())
    query_preview = f"{request.query[:50]}..." if len(request.query) > 50 else request.query

    logger.info(
        f"Received outfit recommendation request | "
        f"trace_id={trace_id} "
        f"user_id={request.user_id} "
        f"session_id={request.session_id} "
        f'query="{query_preview}"'
    )

    try:
        response = await service.recommend(request, trace_id=trace_id)
        logger.info(
            f"Successfully generated outfits | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"outfit_count={len(response.outfits)}"
        )
        return response

    except ParseError as err:
        logger.warning(
            f"Query parsing failed | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"error={err}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"쿼리 파싱 실패: {err}",
        ) from err

    except LLMError as err:
        logger.error(
            f"LLM service error | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"error={err}"
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AI 서비스 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        ) from err

    except Exception as err:
        logger.exception(
            f"Unexpected error during outfit recommendation | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"error={err}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="코디 추천 중 오류가 발생했습니다.",
        ) from err
