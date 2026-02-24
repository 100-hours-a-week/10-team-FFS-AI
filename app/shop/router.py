import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.schemas import ShopSearchRequest, ShopSearchResponse
from app.shop.service import ShopService, get_shop_service

router = APIRouter(prefix="/v2/shop", tags=["shop"])
logger = logging.getLogger(__name__)


@router.post(
    "/outfit",
    response_model=ShopSearchResponse,
    status_code=status.HTTP_200_OK,
)
async def search_shop_outfit(
    request: ShopSearchRequest,
    service: Annotated[ShopService, Depends(get_shop_service)],
) -> ShopSearchResponse:
    trace_id = str(uuid.uuid4())
    query_preview = (
        f"{request.query[:50]}..." if len(request.query) > 50 else request.query
    )

    logger.info(
        f"Received shop search request | "
        f"trace_id={trace_id} "
        f"user_id={request.user_id} "
        f"session_id={request.session_id} "
        f'query="{query_preview}"'
    )

    try:
        response = await service.search(request, trace_id=trace_id)
        logger.info(
            f"Shop search completed | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"outfit_count={len(response.outfits)}"
        )
        return response

    except ShopParseError as err:
        logger.warning(
            f"Shop query parsing failed | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"error={err}"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"쿼리 파싱 실패: {err}",
        ) from err

    except ShopLLMError as err:
        logger.error(
            f"Shop LLM error | "
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
            f"Unexpected shop search error | "
            f"trace_id={trace_id} "
            f"user_id={request.user_id} "
            f"error={err}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="쇼핑 검색 중 오류가 발생했습니다.",
        ) from err
