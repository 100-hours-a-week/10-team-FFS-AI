from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from app.common.schemas import BaseSchema
from app.outfit.schemas import Outfit, UploadSlot


class OutfitRequestMessage(BaseSchema):
    request_id: str = Field(..., description="고유 요청 ID")
    user_id: int = Field(..., description="사용자 ID")
    query: str = Field(..., description="자연어 코디 요청")
    session_id: str = Field(..., description="멀티턴 대화 세션 ID")
    upload_slots: list[UploadSlot] = Field(
        default_factory=list, description="VTON용 업로드 슬롯 정보"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="요청 시각 (UTC)"
    )


class ShopRequestMessage(BaseSchema):
    request_id: str = Field(..., description="고유 요청 ID")
    user_id: int = Field(..., description="사용자 ID")
    query: str = Field(..., description="자연어 쇼핑 요청")
    session_id: str = Field(..., description="멀티턴 대화 세션 ID")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="요청 시각 (UTC)"
    )


class ResponseMetadata(BaseSchema):
    processing_time_ms: int = Field(..., description="처리 시간 (밀리초)")
    model_version: str = Field(default="v1", description="모델 버전")
    shop_supplemented: bool = Field(
        default=False, description="쇼핑 검색으로 아이템 보완 여부"
    )
    fallback_used: bool = Field(default=False, description="Fallback 응답 사용 여부")


class OutfitResponseMessage(BaseSchema):
    request_id: str = Field(..., description="원본 요청 ID")
    status: Literal["success"] = Field(default="success", description="처리 상태")
    outfits: list[Outfit] = Field(default_factory=list, description="추천 코디 목록")
    query_summary: str = Field(default="", description="요청 요약")
    session_id: str | None = Field(default=None, description="세션 ID")
    metadata: ResponseMetadata | None = Field(default=None, description="메타데이터")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="응답 시각 (UTC)"
    )


class ProgressMessage(BaseSchema):
    request_id: str = Field(..., description="요청 ID")
    status: Literal["processing"] = Field(default="processing", description="처리 상태")
    step: int = Field(..., description="현재 단계 번호 (1-based)")
    step_label: str = Field(..., description="단계 레이블 (예: 쿼리 분석 중)")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="이벤트 시각 (UTC)"
    )


class ErrorDetail(BaseSchema):
    code: str = Field(..., description="에러 코드 (예: PARSE_ERROR, LLM_TIMEOUT)")
    message: str = Field(..., description="에러 메시지")
    retry_after_seconds: int | None = Field(
        default=None, description="재시도 권장 대기 시간 (초)"
    )


class ErrorResponse(BaseSchema):
    request_id: str = Field(..., description="원본 요청 ID")
    status: Literal["failed"] = Field(default="failed", description="처리 상태")
    error: ErrorDetail = Field(..., description="에러 상세")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="에러 발생 시각 (UTC)"
    )


class DLQErrorInfo(BaseSchema):
    type: str = Field(..., description="예외 타입 (예: DeserializationError)")
    message: str = Field(..., description="에러 메시지")
    stack_trace: str | None = Field(default=None, description="스택 트레이스")


class DLQMessage(BaseSchema):
    original_topic: str = Field(..., description="원본 토픽 이름")
    original_message: str = Field(..., description="원본 메시지 (문자열)")
    error: DLQErrorInfo = Field(..., description="에러 정보")
    retry_count: int = Field(default=0, description="재시도 횟수")
    failed_at: datetime = Field(
        default_factory=datetime.utcnow, description="실패 시각 (UTC)"
    )
