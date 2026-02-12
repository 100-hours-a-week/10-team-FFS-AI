from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.closet.schemas import AnalyzeRequest, AnalyzeResponse
from app.closet.service import ClosetService, get_closet_service
from app.common.metrics import BATCH_STATUS_REQUESTS

router = APIRouter(prefix="/v1/closet", tags=["closet"])


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=202,
    summary="이미지 분석 요청 (비동기)",
    description="이미지 분석을 요청하고 백그라운드 작업을 시작합니다. 배치 ID와 초기 상태를 반환합니다.",
)
async def analyze_images(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    service: Annotated[ClosetService, Depends(get_closet_service)],
) -> AnalyzeResponse:
    return await service.start_analysis(request, background_tasks)


@router.get(
    "/batches/{batch_id}",
    response_model=AnalyzeResponse,
    summary="분석 배치 상태 조회",
    description="특정 배치 ID의 분석 진행 상태와 결과를 조회합니다.",
)
async def get_batch_status(
    batch_id: str,
    service: Annotated[ClosetService, Depends(get_closet_service)],
) -> AnalyzeResponse:
    BATCH_STATUS_REQUESTS.inc()
    response = await service.get_batch_status(batch_id)
    if not response:
        raise HTTPException(status_code=404, detail={"code": "BATCH_NOT_FOUND"})
    return response
