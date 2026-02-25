"""Closet 분석 요청 핸들러 — Kafka 메시지 처리 파이프라인"""

from __future__ import annotations

import logging

import httpx
from aiokafka import ConsumerRecord

from app.closet.events import (
    AnalyzedPayload,
    AnalyzingCompletedEvent,
    PreprocessingCompletedEvent,
    PreprocessPayload,
    deserialize_request_event,
    serialize_event,
)
from app.closet.gemini_client import GeminiImageAnalyzer
from app.closet.s3_client import S3Client
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.common.metrics import (
    CLOSET_PIPELINE_ERRORS,
    CLOSET_STAGE_DURATION,
    measure_time,
)
from app.config import get_settings
from app.core.kafka import get_kafka_consumer, get_kafka_producer

logger = logging.getLogger(__name__)

_s3_client = S3Client()
_analyzer = GeminiImageAnalyzer()

DEFAULT_ANALYSIS = {
    "major": {
        "category": "UNKNOWN",
        "color": [],
        "material": [],
        "style_tags": [],
    },
    "extra": {
        "meta_data": {
            "gender": None,
            "season": [],
            "formality": None,
            "fit": None,
            "occasion": [],
        },
        "caption": "의류 아이템",
    },
}


async def handle_analysis_request(message: ConsumerRecord) -> None:
    """Kafka 메시지 1건을 전처리 → 분석 → 결과 발행까지 처리합니다."""
    settings = get_settings()
    producer = get_kafka_producer()
    consumer = get_kafka_consumer()

    event = deserialize_request_event(message.value)
    req = event.data
    logger.info(f"요청 수신: batch={req.batch_id}, task={req.task_id}")

    try:
        # 1. 이미지 다운로드
        image_bytes = await _safe_download(req.target_image)
        if image_bytes is None:
            logger.error(f"이미지 다운로드 실패, 건너뜀: task={req.task_id}")
            await consumer.commit()
            return

        # 2. presigned URL 발급
        presigned_info = await _request_presigned_url(
            user_id=req.user_id, purpose="CLOTHES"
        )
        file_id = presigned_info["fileId"]
        presigned_url = presigned_info["presignedUrl"]

        # 3. S3에 이미지 업로드
        await _safe_upload(presigned_url, image_bytes)

        # 4. PREPROCESSING_COMPLETED 이벤트 발행
        preprocess_event = PreprocessingCompletedEvent(
            requested_at=event.requested_at,
            data=PreprocessPayload(
                batchId=req.batch_id,
                taskId=req.task_id,
                fileId=file_id,
            ),
        )
        await producer.send_and_wait(
            settings.kafka_result_topic,
            serialize_event(preprocess_event),
        )
        logger.info(f"전처리 완료: task={req.task_id}")

        # 5. VLM 이미지 분석
        analysis = await _safe_analyze(image_bytes)
        normalized = _normalize_analysis(analysis)

        # 6. ANALYZING_COMPLETED 이벤트 발행
        analyze_event = AnalyzingCompletedEvent(
            requested_at=event.requested_at,
            data=AnalyzedPayload(
                batchId=req.batch_id,
                taskId=req.task_id,
                major=MajorAttributes(**normalized["major"]),
                extra=ExtraAttributes(
                    meta_data=ExtraMetadata(**normalized["extra"]["meta_data"]),
                    caption=normalized["extra"].get("caption"),
                ),
            ),
        )
        await producer.send_and_wait(
            settings.kafka_result_topic,
            serialize_event(analyze_event),
        )
        logger.info(f"분석 완료: task={req.task_id}")

        # 7. 오프셋 커밋
        await consumer.commit()

    except Exception as e:
        logger.error(f"처리 실패: batch={req.batch_id}, task={req.task_id}, error={e}")
        await consumer.commit()


# ── 안전한 단계별 처리 (에러 시 fallback) ──


@measure_time(
    stage="kafka_image_download",
    metric=CLOSET_STAGE_DURATION,
    error_counter=CLOSET_PIPELINE_ERRORS,
)
async def _safe_download(url: str) -> bytes | None:
    """이미지 다운로드. 실패 시 None 반환."""
    try:
        return await _s3_client.get_image(url)
    except Exception as e:
        logger.error(f"다운로드 실패: {e}")
        return None


@measure_time(
    stage="kafka_image_analyze",
    metric=CLOSET_STAGE_DURATION,
    error_counter=CLOSET_PIPELINE_ERRORS,
)
async def _safe_analyze(image_bytes: bytes) -> dict:
    """VLM 분석. 실패 시 DEFAULT_ANALYSIS 반환."""
    try:
        return await _analyzer.analyze_image(image_bytes)
    except Exception as e:
        logger.error(f"분석 실패 (기본값 사용): {e}")
        return DEFAULT_ANALYSIS.copy()


@measure_time(
    stage="kafka_image_upload",
    metric=CLOSET_STAGE_DURATION,
    error_counter=CLOSET_PIPELINE_ERRORS,
)
async def _safe_upload(presigned_url: str, image_bytes: bytes) -> str | None:
    """S3 업로드. 실패 시 에러 문자열 반환."""
    try:
        await _s3_client.put_image(presigned_url, image_bytes)
        return None
    except Exception as e:
        logger.error(f"업로드 실패: {e}")
        return f"UPLOAD_FAILED: {type(e).__name__}"


# ── Presigned URL 발급 ──


async def _request_presigned_url(user_id: int, purpose: str) -> dict:
    """백엔드 내부 API를 호출하여 S3 presigned URL을 발급받습니다."""
    settings = get_settings()
    url = f"{settings.backend_internal_url}/api/internal/presigned-url"

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            url,
            headers={
                "X-Internal-Api-Key": settings.backend_internal_api_key,
                "Content-Type": "application/json",
            },
            json={
                "userId": user_id,
                "purpose": purpose,
                "files": [{"name": "image.jpg", "type": "image/jpeg"}],
            },
        )
        response.raise_for_status()

    result = response.json()
    # 응답: { "code": 201, "data": [{ "fileId": 44189, "objectKey": "...", "presignedUrl": "..." }] }
    return result["data"][0]


# ── 분석 결과 정규화 ──


def _to_list(value: object | None) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value] if value else []


def _normalize_analysis(data: dict) -> dict:
    """VLM 분석 결과를 일관된 형태로 정규화합니다."""
    major = data.get("major", {})
    extra = data.get("extra", {})
    meta = extra.get("meta_data", {})

    return {
        "major": {
            "category": major.get("category") or "UNKNOWN",
            "color": _to_list(major.get("color")),
            "material": _to_list(major.get("material")),
            "style_tags": _to_list(major.get("style_tags")),
        },
        "extra": {
            "meta_data": {
                "gender": meta.get("gender"),
                "season": _to_list(meta.get("season")),
                "formality": meta.get("formality"),
                "fit": meta.get("fit"),
                "occasion": _to_list(meta.get("occasion")),
            },
            "caption": extra.get("caption") or "의류 아이템",
        },
    }
