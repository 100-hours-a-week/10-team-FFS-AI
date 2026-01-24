"""
Closet 모듈 라우터 - API 엔드포인트 정의

API:
- POST /v1/closet/validate: 이미지 어뷰징 체크
- POST /v1/closet/analyze: 이미지 분석 시작
- GET /v1/closet/batches/{batchId}: 분석 상태 조회
"""

from fastapi import APIRouter, HTTPException, status

from app.closet.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BatchStatusResponse,
    ErrorResponse,
    ValidateRequest,
    ValidateResponse,
)
from app.closet.service import (
    get_batch_status,
    start_analyze,
    validate_images,
)

router = APIRouter(prefix="/v1/closet", tags=["closet"])


# ============================================================
# 1. 이미지 어뷰징 체크 API
# ============================================================


@router.post(
    "/validate",
    response_model=ValidateResponse,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청 (필수 필드 누락 등)"},
        422: {
            "model": ErrorResponse,
            "description": "처리 불가 (이미지 개수 제한 위반 등)",
        },
        500: {"model": ErrorResponse, "description": "서버 오류"},
    },
    summary="이미지 어뷰징 체크",
    description="""
    이미지 검증 API (동기)

    검증 항목:
    - 포맷 검증 (jpg, png, webp)
    - 파일 크기 (10MB 이하)
    - 중복/유사 이미지 (Marqo-FashionSigLIP)
    - 패션 도메인 (LAION CLIP)
    - NSFW (Falconsai)
    - 품질 (블러 검출)

    제한사항:
    - 이미지 개수: 1~10개
    """,
)
async def validate(request: ValidateRequest) -> ValidateResponse:
    """이미지 검증 엔드포인트"""
    try:
        return await validate_images(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


# ============================================================
# 2. 이미지 분석 시작 API
# ============================================================


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        400: {"model": ErrorResponse, "description": "잘못된 요청"},
        409: {"model": ErrorResponse, "description": "이미 처리 중"},
        422: {"model": ErrorResponse, "description": "분석할 이미지 없음"},
    },
    summary="이미지 분석 시작",
    description="""
    이미지 분석 시작 API (비동기)

    처리 내용:
    - 배경 제거 (rembg)
    - AI 속성 분석 (카테고리, 색상, 소재, 스타일)

    즉시 202 Accepted 반환 후 백그라운드에서 처리
    """,
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """이미지 분석 시작 엔드포인트"""
    try:
        return start_analyze(request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        ) from e


# ============================================================
# 3. 분석 상태 조회 API (Polling)
# ============================================================


@router.get(
    "/batches/{batch_id}",
    response_model=BatchStatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "배치 없음"},
        410: {"model": ErrorResponse, "description": "만료된 배치"},
    },
    summary="분석 상태 조회",
    description="""
    분석 작업 상태 조회 API (Polling)

    - 권장 폴링 간격: 5초
    - 배치 만료: 24시간 후 삭제
    """,
)
async def get_batch(batch_id: str) -> BatchStatusResponse:
    """배치 상태 조회 엔드포인트"""
    result = get_batch_status(batch_id)

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="배치를 찾을 수 없습니다."
        )

    return result
