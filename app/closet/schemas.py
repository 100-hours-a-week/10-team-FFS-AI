from enum import StrEnum

from pydantic import Field

from app.common.schemas import BaseSchema


class TaskStatus(StrEnum):
    PREPROCESSING = "PREPROCESSING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BatchStatus(StrEnum):
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    PARTIAL_FAILURE = "PARTIAL_FAILURE"


class FileUploadInfo(BaseSchema):
    file_id: int = Field(..., description="파일 ID")
    object_key: str = Field(..., description="S3 오브젝트 키")
    presigned_url: str = Field(..., description="업로드할 Presigned URL")


class AnalyzeImageItem(BaseSchema):
    sequence: int = Field(..., description="순서")
    target_image: str = Field(..., description="분석 대상 이미지 URL (원본)")
    task_id: str = Field(..., description="태스크 ID (UUID)")
    file_upload_info: FileUploadInfo = Field(..., description="업로드 정보")


class AnalyzeRequest(BaseSchema):
    user_id: int = Field(..., description="사용자 ID")
    batch_id: str = Field(..., description="배치 ID (UUID)")
    images: list[AnalyzeImageItem] = Field(..., description="분석할 이미지 목록")


class MajorAttributes(BaseSchema):
    category: str = Field(..., description="카테고리")
    color: list[str] = Field(default_factory=list, description="색상 목록")
    material: list[str] = Field(default_factory=list, description="소재 목록")
    style_tags: list[str] = Field(default_factory=list, description="스타일 태그 목록")


class ExtraMetadata(BaseSchema):
    gender: str | None = Field(default=None, description="성별")
    season: list[str] = Field(default_factory=list, description="계절 목록")
    formality: str | None = Field(default=None, description="격식 수준")
    fit: str | None = Field(default=None, description="핏 (오버핏, 슬림핏 등)")
    occasion: list[str] = Field(default_factory=list, description="상황/장소 목록")


class ExtraAttributes(BaseSchema):
    meta_data: ExtraMetadata = Field(
        default_factory=ExtraMetadata, description="추가 메타데이터"
    )
    caption: str | None = Field(default=None, description="이미지 설명")


class TaskResult(BaseSchema):
    task_id: str = Field(..., description="태스크 ID")
    status: TaskStatus = Field(..., description="태스크 상태")
    file_id: int | None = Field(default=None, description="파일 ID")
    major: MajorAttributes | None = Field(default=None, description="주요 속성")
    extra: ExtraAttributes | None = Field(default=None, description="추가 속성")
    error_message: str | None = Field(default=None, description="에러 메시지")


class BatchMeta(BaseSchema):
    total: int = Field(..., description="전체 요청 수")
    completed: int = Field(..., description="완료된 요청 수")
    processing: int = Field(..., description="처리 중인 요청 수")
    is_finished: bool = Field(..., description="완료 여부")


class AnalyzeResponse(BaseSchema):
    batch_id: str = Field(..., description="배치 ID")
    status: BatchStatus = Field(..., description="배치 상태")
    meta: BatchMeta = Field(..., description="진행 상황 메타데이터")
    results: list[TaskResult] = Field(
        default_factory=list, description="분석 결과 목록"
    )
