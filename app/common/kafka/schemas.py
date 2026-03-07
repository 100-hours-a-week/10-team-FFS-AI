from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone





class OutfitMetadata(BaseModel):

    confidence: float
    shop_supplemented: bool = False
    fallback_used: bool = False
    processing_time_ms: int



class OutfitRequestMessage(BaseModel):

    request_id: str = Field(..., description="요청 추적용 고유 ID (UUID)")
    user_id: int
    query: str
    session_id: Optional[str] = None
    upload_slots: Optional[List[str]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class ShopRequestMessage(BaseModel):

    request_id: str
    user_id: int
    query: str
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class OutfitResponseMessage(BaseModel):

    request_id: str
    status: Literal["success"] = "success"
    outfits: List[Dict[str, Any]]
    metadata: OutfitMetadata
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorDetail(BaseModel):
    code: str
    message: str
    retry_after_seconds: Optional[int] = 60


class DLQErrorDetail(BaseModel):
    type: str
    message: str
    stack_trace: Optional[str] = None

class ErrorResponse(BaseModel):

    request_id: str
    status: Literal["failed"] = "failed"
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DLQMessage(BaseModel):

    original_topic: str
    original_message: Dict[str, Any]
    error: DLQErrorDetail
    retry_count: int
    failed_at: datetime = Field(default_factory=datetime.utcnow)