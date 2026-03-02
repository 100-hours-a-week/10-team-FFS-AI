"""Kafka 이벤트 스키마 — 백엔드 ↔ AI 서버 메시지 정의"""

from __future__ import annotations

import json
import logging
from enum import StrEnum

from pydantic import Field

from app.closet.schemas import ExtraAttributes, MajorAttributes
from app.common.schemas import BaseSchema

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    ANALYSIS_REQUESTED = "AI_ANALYSIS_REQUESTED"
    ABUSING_COMPLETED = "AI_ANALYSIS_ABUSING_COMPLETED"
    SEGMENTATION_COMPLETED = "AI_ANALYSIS_SEGMENTATION_COMPLETED"
    PREPROCESSING_COMPLETED = "AI_ANALYSIS_PREPROCESSING_COMPLETED"
    ANALYZING_COMPLETED = "AI_ANALYSIS_ANALYZING_COMPLETED"


# ── 수신 이벤트: 백엔드 → AI 서버 ──


class RequestPayload(BaseSchema):
    batch_id: str = Field(..., alias="batchId")
    source_id: str = Field(..., alias="sourceId")
    user_id: int = Field(..., alias="userId")
    target_image: str = Field(..., alias="targetImage")


class AnalysisRequestedEvent(BaseSchema):
    """Kafka 요청 토픽에서 수신하는 분석 요청 이벤트"""

    event_type: str = Field(..., alias="eventType")
    requested_at: str = Field(..., alias="requestedAt")
    data: RequestPayload


# ── 발행 이벤트 (0a): 어뷰징 검사 완료 ──


class AbusingPayload(BaseSchema):
    batch_id: str = Field(..., alias="batchId")
    source_id: str = Field(..., alias="sourceId")
    passed: bool


class AbusingCompletedEvent(BaseSchema):
    """어뷰징 검사 완료 이벤트 (항상 passed=true, 실제 로직은 추후 구현)"""

    event_type: str = Field(default=EventType.ABUSING_COMPLETED, alias="eventType")
    requested_at: str = Field(..., alias="requestedAt")
    data: AbusingPayload


# ── 발행 이벤트 (0b): 세그멘테이션 완료 ──


class SegmentationPayload(BaseSchema):
    batch_id: str = Field(..., alias="batchId")
    source_id: str = Field(..., alias="sourceId")
    segmentation: int = Field(..., alias="segmentation")


class SegmentationCompletedEvent(BaseSchema):
    """세그멘테이션 완료 이벤트 — 아이템 개수(totalItems)를 백엔드에 알림"""

    event_type: str = Field(default=EventType.SEGMENTATION_COMPLETED, alias="eventType")
    requested_at: str = Field(..., alias="requestedAt")
    data: SegmentationPayload


# ── 발행 이벤트 (1): 전처리 완료 ──


class PreprocessPayload(BaseSchema):
    batch_id: str = Field(..., alias="batchId")
    source_id: str = Field(..., alias="sourceId")
    task_id: str = Field(..., alias="taskId")
    file_id: int = Field(..., alias="fileId")


class PreprocessingCompletedEvent(BaseSchema):
    """이미지 다운로드 + S3 업로드 완료 시 결과 토픽에 발행"""

    event_type: str = Field(
        default=EventType.PREPROCESSING_COMPLETED, alias="eventType"
    )
    requested_at: str = Field(..., alias="requestedAt")
    data: PreprocessPayload


# ── 발행 이벤트 (2): 분석 완료 ──


class AnalyzedPayload(BaseSchema):
    batch_id: str = Field(..., alias="batchId")
    source_id: str = Field(..., alias="sourceId")
    task_id: str = Field(..., alias="taskId")
    major: MajorAttributes
    extra: ExtraAttributes


class AnalyzingCompletedEvent(BaseSchema):
    """분석 완료 시 결과 토픽에 발행 (major/extra JSON 포함)"""

    event_type: str = Field(default=EventType.ANALYZING_COMPLETED, alias="eventType")
    requested_at: str = Field(..., alias="requestedAt")
    data: AnalyzedPayload


# ── 직렬화 / 역직렬화 ──


def serialize_event(event: BaseSchema) -> bytes:
    """Pydantic 모델 → JSON bytes (by_alias=True: snake_case → camelCase)"""
    return event.model_dump_json(by_alias=True).encode("utf-8")


def deserialize_request_event(raw: bytes) -> AnalysisRequestedEvent:
    """JSON bytes → AnalysisRequestedEvent 모델"""
    try:
        data = json.loads(raw.decode("utf-8"))
        return AnalysisRequestedEvent.model_validate(data)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"이벤트 역직렬화 실패: {e}, 원본: {raw[:200]}")
        raise
