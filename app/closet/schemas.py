"""
Closet 모듈 스키마 정의
- ValidateRequest/Response: 이미지 검증 API
- AnalyzeRequest/Response: 이미지 분석 시작 API
- TaskStatusResponse: 분석 상태 조회 API
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# 공통 Enum 정의
# ============================================================

class ValidationStatus(str, Enum):
    """검증 결과 상태"""
    PASSED = "PASSED"
    FAILED = "FAILED"


class ValidationErrorCode(str, Enum):
    """검증 에러 코드"""
    INVALID_FORMAT = "INVALID_FORMAT"       # 지원하지 않는 이미지 포맷
    FILE_TOO_LARGE = "FILE_TOO_LARGE"       # 파일 크기 10MB 초과
    EXACT_DUPLICATE = "EXACT_DUPLICATE"     # 완전 동일 이미지 존재
    SIMILAR_ITEM = "SIMILAR_ITEM"           # 유사 아이템 존재 (similarity ≥ 0.95)
    NOT_FASHION = "NOT_FASHION"             # 패션 아이템 아님
    NSFW_DETECTED = "NSFW_DETECTED"         # 부적절 콘텐츠 감지
    TOO_BLURRY = "TOO_BLURRY"               # 이미지 품질 불량


class TaskStatus(str, Enum):
    """작업 상태"""
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ProgressStatus(str, Enum):
    """진행 상태"""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    DONE = "DONE"
    FAILED = "FAILED"


# ============================================================
# 1. Validate API - POST /v1/closet/validate
# ============================================================

class ValidateRequest(BaseModel):
    """이미지 검증 요청"""
    userId: str = Field(..., description="사용자 고유 식별자 (UUID 권장)")
    images: list[str] = Field(
        ..., 
        description="검증할 이미지 URL 목록",
        max_length=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "userId": "user12345",
                "images": [
                    "https://s3.example.com/temp/image1.jpg",
                    "https://s3.example.com/temp/image2.jpg"
                ]
            }
        }


class ValidationResult(BaseModel):
    """개별 이미지 검증 결과"""
    originUrl: str = Field(..., description="원본 이미지 URL")
    status: ValidationStatus = Field(..., description="검증 결과 (PASSED/FAILED)")
    errorCode: Optional[ValidationErrorCode] = Field(
        None, 
        description="실패 시 에러 코드"
    )


class ValidationSummary(BaseModel):
    """검증 결과 요약"""
    total: int = Field(..., description="전체 이미지 수")
    passed: int = Field(..., description="통과한 이미지 수")
    failed: int = Field(..., description="실패한 이미지 수")


class ValidateResponse(BaseModel):
    """이미지 검증 응답"""
    success: bool = Field(..., description="요청 성공 여부")
    validationId: str = Field(..., description="검증 ID")
    processingTimeMs: int = Field(..., description="처리 시간 (ms)")
    validationSummary: ValidationSummary = Field(..., description="검증 결과 요약")
    validationResults: list[ValidationResult] = Field(..., description="개별 검증 결과")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "validationId": "val_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "processingTimeMs": 1200,
                "validationSummary": {
                    "total": 10,
                    "passed": 7,
                    "failed": 3
                },
                "validationResults": [
                    {"originUrl": "https://s3.example.com/temp/image1.jpg", "status": "PASSED"},
                    {"originUrl": "https://s3.example.com/temp/image2.jpg", "status": "FAILED", "errorCode": "DUPLICATE"}
                ]
            }
        }


# ============================================================
# 2. Analyze API - POST /v1/closet/analyze
# ============================================================

class AnalyzeImageItem(BaseModel):
    """분석할 개별 이미지 정보"""
    originUrl: str = Field(..., description="원본 이미지 URL")
    uploadUrl: str = Field(..., description="S3 업로드된 이미지 URL")
    imageKey: str = Field(..., description="S3 이미지 키")
    sequence: int = Field(..., description="이미지 순서", ge=0)


class AnalyzeRequest(BaseModel):
    """이미지 분석 시작 요청"""
    userId: str = Field(..., description="사용자 식별자")
    images: list[AnalyzeImageItem] = Field(..., description="분석할 이미지 목록 (검증 통과된 것만)")

    class Config:
        json_schema_extra = {
            "example": {
                "userId": "user_12345",
                "images": [
                    {
                        "originUrl": "https://s3.example.com/temp/image1.jpg",
                        "uploadUrl": "https://s3.example.com/uploads/user_12345/image1.jpg",
                        "imageKey": "uploads/user_12345/image1.jpg",
                        "sequence": 0
                    }
                ]
            }
        }


class TaskIdItem(BaseModel):
    """개별 이미지의 작업 ID"""
    originUrl: str = Field(..., description="원본 이미지 URL")
    taskId: str = Field(..., description="작업 식별자")


class AnalyzeResponse(BaseModel):
    """이미지 분석 시작 응답 (202 Accepted)"""
    taskId: str = Field(..., description="전체 작업 식별자")
    status: TaskStatus = Field(default=TaskStatus.PROCESSING, description="작업 상태")
    queued: int = Field(..., description="대기 중인 이미지 수")
    taskIds: list[TaskIdItem] = Field(..., description="개별 이미지 작업 ID 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "taskId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "status": "PROCESSING",
                "queued": 7,
                "taskIds": [
                    {"originUrl": "https://s3.example.com/temp/image1.jpg", "taskId": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
                ]
            }
        }


# ============================================================
# 3. Task Status API - GET /v1/closet/tasks/{taskId}
# ============================================================

class ProgressInfo(BaseModel):
    """개별 이미지 진행 상태"""
    backgroundRemoval: ProgressStatus = Field(..., description="배경 제거 진행 상태")
    analysis: ProgressStatus = Field(..., description="AI 분석 진행 상태")


class BackgroundRemovalResult(BaseModel):
    """배경 제거 결과"""
    imageKey: str = Field(..., description="처리된 이미지 S3 키")


class AnalysisAttributes(BaseModel):
    """AI 분석 속성 결과"""
    category: str = Field(..., description="카테고리 (상의, 하의, 아우터 등)")
    color: list[str] = Field(..., description="색상 목록")
    material: list[str] = Field(..., description="소재 목록")
    styleTags: list[str] = Field(..., description="스타일 태그 목록")


class AnalysisResult(BaseModel):
    """AI 분석 결과"""
    attributes: AnalysisAttributes = Field(..., description="분석된 속성들")


class TaskResultItem(BaseModel):
    """개별 이미지 작업 결과"""
    taskId: str = Field(..., description="작업 식별자")
    originUrl: str = Field(..., description="원본 이미지 URL")
    status: TaskStatus = Field(..., description="작업 상태")
    progress: ProgressInfo = Field(..., description="진행 상태")
    backgroundRemoval: Optional[BackgroundRemovalResult] = Field(
        None, 
        description="배경 제거 결과 (완료 시)"
    )
    analysis: Optional[AnalysisResult] = Field(
        None, 
        description="AI 분석 결과 (완료 시)"
    )


class TaskMeta(BaseModel):
    """작업 메타 정보"""
    total: int = Field(..., description="전체 이미지 수")
    completed: int = Field(..., description="완료된 이미지 수")
    processing: int = Field(..., description="처리 중인 이미지 수")
    isFinished: bool = Field(..., description="전체 작업 완료 여부")


class TaskStatusResponse(BaseModel):
    """작업 상태 조회 응답"""
    taskId: str = Field(..., description="전체 작업 식별자")
    status: TaskStatus = Field(..., description="전체 작업 상태")
    meta: TaskMeta = Field(..., description="작업 메타 정보")
    results: list[TaskResultItem] = Field(..., description="개별 이미지 결과 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "taskId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "status": "PROCESSING",
                "meta": {
                    "total": 7,
                    "completed": 3,
                    "processing": 4,
                    "isFinished": False
                },
                "results": [
                    {
                        "taskId": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
                        "originUrl": "https://s3.example.com/temp/image1.jpg",
                        "status": "COMPLETED",
                        "progress": {"backgroundRemoval": "DONE", "analysis": "DONE"},
                        "backgroundRemoval": {"imageKey": "uploads/user_12345/image1.jpg"},
                        "analysis": {
                            "attributes": {
                                "category": "상의",
                                "color": ["빨강"],
                                "material": ["니트"],
                                "styleTags": ["캐주얼", "따뜻한"]
                            }
                        }
                    }
                ]
            }
        }


# ============================================================
# 에러 응답 스키마
# ============================================================

class ErrorResponse(BaseModel):
    """공통 에러 응답"""
    success: bool = Field(default=False)
    errorCode: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    detail: Optional[str] = Field(None, description="상세 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "errorCode": "INVALID_FORMAT",
                "message": "지원하지 않는 이미지 포맷입니다.",
                "detail": "jpg, png, webp 형식만 지원합니다."
            }
        }
