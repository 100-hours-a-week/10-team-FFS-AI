from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import Field

from app.common.schemas import BaseSchema


class OutfitMetadata(BaseSchema):
    confidence: float
    shop_supplemented: bool = False
    fallback_used: bool = False
    processing_time_ms: int


class OutfitRequestMessage(BaseSchema):
    request_id: str = Field(..., description="요청 추적용 고유 ID (UUID)")
    user_id: int
    query: str
    session_id: str | None = None
    upload_slots: list[str] | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ShopRequestMessage(BaseSchema):
    request_id: str
    user_id: int
    query: str
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class OutfitResponseMessage(BaseSchema):
    request_id: str
    status: Literal["success"] = "success"
    outfits: list[dict[str, Any]]
    metadata: OutfitMetadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ErrorDetail(BaseSchema):
    code: str
    message: str
    retry_after_seconds: int | None = 60


class DLQErrorDetail(BaseSchema):
    type: str
    message: str
    stack_trace: str | None = None


class ErrorResponse(BaseSchema):
    request_id: str
    status: Literal["failed"] = "failed"
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class DLQMessage(BaseSchema):
    original_topic: str
    original_message: dict[str, Any]
    error: DLQErrorDetail
    retry_count: int
    failed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ProgressMessage(BaseSchema):
    request_id: str
    status: str
    step: str
    step_label: str
    timestamp: float
