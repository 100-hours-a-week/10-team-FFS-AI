"""Closet 비즈니스 로직 서비스 — 이미지 분석 파이프라인

이 모듈은 통신(Kafka/HTTP) 방식에 독립적인 순수 비즈니스 로직만 담당합니다.
- S3에서 이미지 다운로드
- 백엔드 내부 API로 Presigned URL 발급
- S3에 이미지 업로드
- Gemini로 이미지 분석
- 분석 결과 정규화
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

from app.closet.gemini_client import GeminiImageAnalyzer
from app.closet.s3_client import S3Client
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.common.metrics import (
    CLOSET_PIPELINE_ERRORS,
    CLOSET_STAGE_DURATION,
    measure_time,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS: dict = {
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


@dataclass
class PreprocessResult:
    """전처리(다운로드 + S3 업로드) 결과를 담는 데이터 클래스."""

    file_id: int
    success: bool
    error: str | None = None


@dataclass
class AnalysisResult:
    """이미지 분석 결과를 담는 데이터 클래스."""

    major: MajorAttributes
    extra: ExtraAttributes
    success: bool
    error: str | None = None


class ClosetService:
    """통신 방식에 독립적인 옷 이미지 분석 비즈니스 로직."""

    def __init__(self) -> None:
        self._s3_client = S3Client()
        self._analyzer = GeminiImageAnalyzer()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def preprocess(self, target_image_url: str, user_id: int) -> PreprocessResult:
        """이미지를 다운로드하고 S3에 업로드합니다(전처리 단계).

        Returns:
            PreprocessResult: file_id와 성공 여부를 담은 결과.
        """
        # 1. 이미지 다운로드
        image_bytes = await self._safe_download(target_image_url)
        if image_bytes is None:
            return PreprocessResult(
                file_id=0, success=False, error="IMAGE_DOWNLOAD_FAILED"
            )

        # 2. Presigned URL 발급
        presigned_info = await self._request_presigned_url(
            user_id=user_id, purpose="CLOTHES"
        )
        file_id = presigned_info["fileId"]
        presigned_url = presigned_info["presignedUrl"]

        # 3. S3에 이미지 업로드
        upload_error = await self._safe_upload(presigned_url, image_bytes)
        if upload_error:
            return PreprocessResult(file_id=file_id, success=False, error=upload_error)

        return PreprocessResult(file_id=file_id, success=True)

    async def analyze(self, target_image_url: str) -> AnalysisResult:
        """이미지를 다운로드하고 분석합니다(분석 단계).

        Returns:
            AnalysisResult: 분석된 major/extra 속성과 성공 여부를 담은 결과.
        """
        # 1. 이미지 다운로드
        image_bytes = await self._safe_download(target_image_url)
        if image_bytes is None:
            normalized = DEFAULT_ANALYSIS.copy()
            return AnalysisResult(
                major=MajorAttributes(**normalized["major"]),
                extra=ExtraAttributes(
                    meta_data=ExtraMetadata(**normalized["extra"]["meta_data"]),
                    caption=normalized["extra"].get("caption"),
                ),
                success=False,
                error="IMAGE_DOWNLOAD_FAILED",
            )

        # 2. 이미지 분석
        raw_analysis = await self._safe_analyze(image_bytes)
        normalized = self._normalize_analysis(raw_analysis)

        return AnalysisResult(
            major=MajorAttributes(**normalized["major"]),
            extra=ExtraAttributes(
                meta_data=ExtraMetadata(**normalized["extra"]["meta_data"]),
                caption=normalized["extra"].get("caption"),
            ),
            success=True,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 안전 래퍼 (에러 시 fallback)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @measure_time(
        stage="image_download",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def _safe_download(self, url: str) -> bytes | None:
        """이미지 다운로드. 실패 시 None 반환."""
        try:
            return await self._s3_client.get_image(url)
        except Exception as e:
            logger.error(f"다운로드 실패: {e}")
            return None

    @measure_time(
        stage="image_analyze",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def _safe_analyze(self, image_bytes: bytes) -> dict:
        """이미지 분석. 실패 시 DEFAULT_ANALYSIS 반환."""
        try:
            return await self._analyzer.analyze_image(image_bytes)
        except Exception as e:
            logger.error(f"분석 실패 (기본값 사용): {e}")
            return DEFAULT_ANALYSIS.copy()

    @measure_time(
        stage="image_upload",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def _safe_upload(self, presigned_url: str, image_bytes: bytes) -> str | None:
        """S3 업로드. 실패 시 에러 문자열 반환."""
        try:
            await self._s3_client.put_image(presigned_url, image_bytes)
            return None
        except Exception as e:
            logger.error(f"업로드 실패: {e}")
            return f"UPLOAD_FAILED: {type(e).__name__}"

    async def _request_presigned_url(self, user_id: int, purpose: str) -> dict:
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

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 분석 결과 정규화
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _to_list(value: object | None) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value] if value else []

    @classmethod
    def _normalize_analysis(cls, data: dict) -> dict:
        """분석 결과를 일관된 형태로 정규화합니다."""
        major = data.get("major", {})
        extra = data.get("extra", {})
        meta = extra.get("meta_data", {})

        return {
            "major": {
                "category": major.get("category") or "UNKNOWN",
                "color": cls._to_list(major.get("color")),
                "material": cls._to_list(major.get("material")),
                "style_tags": cls._to_list(major.get("style_tags")),
            },
            "extra": {
                "meta_data": {
                    "gender": meta.get("gender"),
                    "season": cls._to_list(meta.get("season")),
                    "formality": meta.get("formality"),
                    "fit": meta.get("fit"),
                    "occasion": cls._to_list(meta.get("occasion")),
                },
                "caption": extra.get("caption") or "의류 아이템",
            },
        }
