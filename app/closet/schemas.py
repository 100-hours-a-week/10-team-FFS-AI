"""
Closet 모듈 스키마 정의 (API 명세 v2 기준)

담당 API:
- ValidateRequest/Response: 이미지 어뷰징 체크 API (2.1)
- AnalyzeRequest/Response: 이미지 분석 시작 API (2.2)
- BatchStatusResponse: 분석 상태 조회 API (2.3)
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

# ============================================================
# 공통 Enum 정의
# ============================================================


class ValidationErrorCode(str, Enum):
    """검증 에러 코드"""

    INVALID_FORMAT = "INVALID_FORMAT"  # 지원하지 않는 이미지 포맷
    FILE_TOO_LARGE = "FILE_TOO_LARGE"  # 파일 크기 10MB 초과
    DUPLICATE = "DUPLICATE"  # 동일/유사 이미지 존재
    NOT_FASHION = "NOT_FASHION"  # 패션 아이템 아님
    NSFW = "NSFW"  # 부적절 콘텐츠 감지
    TOO_BLURRY = "TOO_BLURRY"  # 이미지 품질 불량


class BatchStatus(str, Enum):
    """배치 작업 상태"""

    ACCEPTED = "ACCEPTED"  # 작업 수락됨 (analyze 응답용)
    IN_PROGRESS = "IN_PROGRESS"  # 실행 중
    COMPLETED = "COMPLETED"  # 완료
    FAILED = "FAILED"  # 실패


class TaskStatus(str, Enum):
    """개별 작업 상태"""

    PREPROCESSING = "PREPROCESSING"  # 배경 제거 진행 중
    ANALYZING = "ANALYZING"  # AI 분석 진행 중
    COMPLETED = "COMPLETED"  # 완료
    FAILED = "FAILED"  # 실패


# ============================================================
# 1. Validate API - POST /v1/closet/validate
# ============================================================


class ValidateRequest(BaseModel):
    """이미지 검증 요청"""

    userId: int = Field(..., description="사용자 ID")
    images: list[str] = Field(
        ..., description="검증할 이미지 URL 목록 (1~10개)", min_length=1, max_length=10
    )

    class Config:
        json_schema_extra = {
            "example": {
                "userId": 123,
                "images": [
                    "https://s3.example.com/image1.jpg",
                    "https://s3.example.com/image2.jpg",
                    "https://s3.example.com/image3.jpg",
                ],
            }
        }


class ValidationResult(BaseModel):
    """개별 이미지 검증 결과"""

    originUrl: str = Field(..., description="원본 이미지 URL")
    passed: bool = Field(..., description="검증 통과 여부")
    error: Optional[ValidationErrorCode] = Field(None, description="실패 시 에러 코드")


class ValidationSummary(BaseModel):
    """검증 결과 요약"""

    total: int = Field(..., description="전체 이미지 수")
    passed: int = Field(..., description="통과한 이미지 수")
    failed: int = Field(..., description="실패한 이미지 수")


class ValidateResponse(BaseModel):
    """이미지 검증 응답"""

    success: bool = Field(..., description="요청 성공 여부")
    validationSummary: ValidationSummary = Field(..., description="검증 결과 요약")
    validationResults: list[ValidationResult] = Field(..., description="개별 검증 결과")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "validationSummary": {"total": 5, "passed": 3, "failed": 2},
                "validationResults": [
                    {"originUrl": "https://s3.example.com/image1.jpg", "passed": True},
                    {"originUrl": "https://s3.example.com/image2.jpg", "passed": True},
                    {"originUrl": "https://s3.example.com/image3.jpg", "passed": True},
                    {
                        "originUrl": "https://s3.example.com/image4.jpg",
                        "passed": False,
                        "error": "DUPLICATE",
                    },
                    {
                        "originUrl": "https://s3.example.com/image5.jpg",
                        "passed": False,
                        "error": "NSFW",
                    },
                ],
            }
        }


# ============================================================
# 2. Analyze API - POST /v1/closet/analyze
# ============================================================


class FileInfo(BaseModel):
    """파일 정보"""

    fileId: int = Field(..., description="파일 ID")
    objectKey: str = Field(..., description="S3 객체 키")
    presignedUrl: str = Field(..., description="S3 presigned URL")


class AnalyzeImageItem(BaseModel):
    """분석할 개별 이미지 정보"""

    sequence: int = Field(..., description="이미지 순서", ge=0)
    targetImage: str = Field(..., description="원본 이미지 URL")
    taskId: str = Field(..., description="개별 작업 ID (UUID)")
    fileUploadInfo: FileInfo = Field(..., description="파일 업로드 정보")


class AnalyzeRequest(BaseModel):
    """이미지 분석 시작 요청"""

    userId: int = Field(..., description="사용자 ID")
    batchId: str = Field(..., description="배치 작업 ID (UUID)")
    images: list[AnalyzeImageItem] = Field(..., description="분석할 이미지 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "userId": 123,
                "batchId": "123e4567-e89b-12d3-a456-426614174000",
                "images": [
                    {
                        "sequence": 0,
                        "targetImage": "https://s3.example.com/image1.jpg",
                        "taskId": "123e4567-e89b-12d3-a456-426614174001",
                        "fileUploadInfo": {
                            "fileId": 45,
                            "objectKey": "sample1.jpg",
                            "presignedUrl": "https://s3.example.com/sample-upload1.jpg",
                        },
                    },
                    {
                        "sequence": 1,
                        "targetImage": "https://s3.example.com/image2.jpg",
                        "taskId": "123e4567-e89b-12d3-a456-426614174002",
                        "fileUploadInfo": {
                            "fileId": 46,
                            "objectKey": "sample2.jpg",
                            "presignedUrl": "https://s3.example.com/sample-upload2.jpg",
                        },
                    },
                ],
            }
        }


class BatchMeta(BaseModel):
    """배치 작업 메타 정보"""

    total: int = Field(..., description="전체 이미지 수")
    completed: int = Field(..., description="완료된 이미지 수")
    processing: int = Field(..., description="처리 중인 이미지 수")
    isFinished: bool = Field(..., description="전체 작업 완료 여부")


class AnalyzeResponse(BaseModel):
    """이미지 분석 시작 응답 (202 Accepted)"""

    batchId: str = Field(..., description="배치 작업 ID")
    status: BatchStatus = Field(default=BatchStatus.ACCEPTED, description="상태")
    meta: BatchMeta = Field(..., description="메타 정보")

    class Config:
        json_schema_extra = {
            "example": {
                "batchId": "123e4567-e89b-12d3-a456-426614174000",
                "status": "ACCEPTED",
                "meta": {
                    "total": 3,
                    "completed": 0,
                    "processing": 3,
                    "isFinished": False,
                },
            }
        }


# ============================================================
# 3. Batch Status API - GET /v1/closet/batches/{batchId}
# ============================================================


class AnalysisAttributes(BaseModel):
    """AI 분석 속성 결과"""

    category: str = Field(..., description="카테고리 (상의, 하의, 아우터 등)")
    color: list[str] = Field(..., description="색상 목록")
    material: list[str] = Field(..., description="소재 목록")
    styleTags: list[str] = Field(..., description="스타일 태그 목록")


class BatchResultItem(BaseModel):
    """개별 이미지 작업 결과"""

    taskId: str = Field(..., description="개별 작업 ID")
    status: TaskStatus = Field(
        ..., description="작업 상태 (PREPROCESSING/ANALYZING/COMPLETED/FAILED)"
    )
    fileId: Optional[int] = Field(None, description="처리된 파일 ID (전처리 완료 시)")
    attributes: Optional[AnalysisAttributes] = Field(
        None, description="AI 분석 속성 (완료 시)"
    )


class BatchStatusResponse(BaseModel):
    """배치 상태 조회 응답"""

    batchId: str = Field(..., description="배치 작업 ID")
    status: BatchStatus = Field(
        ..., description="배치 상태 (IN_PROGRESS/COMPLETED/FAILED)"
    )
    meta: BatchMeta = Field(..., description="메타 정보")
    results: list[BatchResultItem] = Field(..., description="개별 작업 결과 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "batchId": "123e4567-e89b-12d3-a456-426614174000",
                "status": "IN_PROGRESS",
                "meta": {
                    "total": 3,
                    "completed": 1,
                    "processing": 2,
                    "isFinished": False,
                },
                "results": [
                    {
                        "taskId": "123e4567-e89b-12d3-a456-426614174001",
                        "status": "PREPROCESSING",
                    },
                    {
                        "taskId": "123e4567-e89b-12d3-a456-426614174002",
                        "status": "ANALYZING",
                        "fileId": 46,
                    },
                    {
                        "taskId": "123e4567-e89b-12d3-a456-426614174003",
                        "status": "COMPLETED",
                        "fileId": 47,
                        "attributes": {
                            "category": "상의",
                            "color": ["빨강"],
                            "material": ["니트"],
                            "styleTags": ["캐주얼", "따뜻한"],
                        },
                    },
                ],
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

    class Config:
        json_schema_extra = {
            "example": {
                "success": False,
                "errorCode": "INVALID_FORMAT",
                "message": "지원하지 않는 이미지 포맷입니다.",
            }
        }
