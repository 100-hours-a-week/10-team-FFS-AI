import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.closet.router import router as closet_router
from app.core.database import check_health, close_databases, init_databases
from app.embedding.router import router as embedding_router
from app.outfit.router import router as outfit_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    logger.info("start server")
    try:
        await init_databases()
        logger.info("Application startup complete")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

    yield

    logger.info("shut down server")
    try:
        await close_databases()
        logger.info("Application shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


app = FastAPI(
    title="KlosetLab",
    lifespan=lifespan,
)

app.include_router(embedding_router, prefix="/ai")
app.include_router(outfit_router, prefix="/ai")
app.include_router(closet_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"status": "running"}


@app.get("/health")
async def health_check() -> JSONResponse:
    health_status = await check_health()

    all_connected = all(status == "connected" for status in health_status.values())
    overall_status = "healthy" if all_connected else "degraded"

    response_data = {
        "status": overall_status,
        "services": health_status,
    }
    status_code = 200 if all_connected else 503

    return JSONResponse(content=response_data, status_code=status_code)


# ============================================================
# 커스텀 에러 핸들러
# ============================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Pydantic 검증 에러 핸들러

    - 필수 필드 누락 → 400 Bad Request
    - 비즈니스 규칙 위반 (min_length, max_length 등) → 422 Unprocessable Entity
    """
    errors = exc.errors()
    first_error = errors[0] if errors else {}
    error_type = first_error.get("type", "")
    loc = first_error.get("loc", [])

    # 필드명 추출 (body, userId -> userId)
    field_name = loc[-1] if loc else ""

    # 에러 타입과 필드에 따른 메시지 매핑
    message = _get_error_message(error_type, field_name)

    # 400: 필수 필드 누락, 타입 에러 (잘못된 요청)
    # 422: 비즈니스 규칙 위반 (개수 제한 등)
    if "missing" in error_type or "type" in error_type:
        status_code = status.HTTP_400_BAD_REQUEST
        error_code = "INVALID_REQUEST"
    else:
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
        error_code = "VALIDATION_ERROR"

    return JSONResponse(
        status_code=status_code,
        content={"success": False, "errorCode": error_code, "message": message},
    )


def _get_error_message(error_type: str, field_name: str) -> str:
    """에러 타입과 필드명에 따른 한글 메시지 반환"""

    # 필수 필드 누락 (400)
    if "missing" in error_type:
        field_messages = {
            "userId": "userId가 누락됐습니다",
            "images": "images가 누락됐습니다",
            "batchId": "batchId가 누락됐습니다",
            "taskId": "taskId가 누락됐습니다",
        }
        return field_messages.get(field_name, f"{field_name}가 누락됐습니다")

    # 타입 에러 (400) - 배열이 아닌 경우 등
    if "type" in error_type:
        if field_name == "images":
            return "images는 URL 배열이어야 합니다"
        return f"{field_name}의 타입이 올바르지 않습니다"

    # 리스트 길이 검증 (422) - 통일된 메시지
    if "too_short" in error_type or "too_long" in error_type:
        if field_name == "images":
            return "이미지는 1개 이상 10개 이하여야 합니다"
        return f"{field_name}의 개수가 올바르지 않습니다"

    # 기본 메시지
    return "요청 데이터가 올바르지 않습니다"
