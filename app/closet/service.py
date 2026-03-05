"""Closet 비즈니스 로직 서비스 — 이미지 분석 파이프라인

이 모듈은 통신(Kafka/HTTP) 방식에 독립적인 순수 비즈니스 로직만 담당합니다.
- 세그멘테이션: 모델 착용 이미지 → N개 개별 아이템 추출
- 백엔드 내부 API로 Presigned URL 발급 (N개)
- S3에 이미지 업로드 (N개 병렬)
- Gemini로 이미지 분석
- 분석 결과 정규화
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import httpx
from ulid import ULID

from app.closet.gemini_client import GeminiImageAnalyzer
from app.closet.s3_client import S3Client
from app.closet.schemas import ExtraAttributes, ExtraMetadata, MajorAttributes
from app.closet.validators import ImageValidator, MockImageValidator
from app.common.metrics import (
    CLOSET_PIPELINE_ERRORS,
    CLOSET_STAGE_DURATION,
    measure_time,
)
from app.config import get_settings

logger = logging.getLogger(__name__)

DEFAULT_ANALYSIS: dict = {
    "major": {
        "category": "ETC",
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
    """전처리(세그멘테이션 + S3 업로드) 결과를 담는 데이터 클래스."""

    file_ids: list[int] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    item_images: list[bytes] = field(default_factory=list)
    success: bool = False
    error: str | None = None


@dataclass
class AnalysisResult:
    """이미지 분석 결과를 담는 데이터 클래스."""

    major: MajorAttributes
    extra: ExtraAttributes
    success: bool
    error: str | None = None


class ClosetService:
    """Closet 분석 비즈니스 로직 — 통신 방식에 독립적."""

    def __init__(self) -> None:
        settings = get_settings()
        self._s3_client = S3Client()
        self._use_mock = settings.use_mock_analyzer

        self._fallback_analyzer = None

        if self._use_mock:
            from app.closet.mock_analyzer import (
                MockImageAnalyzer,
                MockSegmentationService,
            )

            self._analyzer = MockImageAnalyzer(delay_seconds=4.0)
            self._segmentation = MockSegmentationService(
                item_count=3, delay_seconds=9.0
            )
            self.validator = MockImageValidator()
            logger.info("Using Mock mode for load testing")
        elif settings.vllm_server_url:
            from app.closet.model_analyzer import ModelServerAnalyzer

            self._analyzer = ModelServerAnalyzer(
                base_url=settings.vllm_server_url,
            )
            self._fallback_analyzer = GeminiImageAnalyzer()
            from app.closet.segmentation import SegmentationService

            self._segmentation = SegmentationService(
                gemini_client=self._fallback_analyzer,
            )
            self.validator = ImageValidator()
            logger.info(
                f"Using vLLM model server: {settings.vllm_server_url} "
                "(Gemini fallback enabled)"
            )
        else:
            logger.warning(
                "GCP_SERVER_URL environment variable is not set. "
                "Falling back to Gemini API for image analysis."
            )
            self._analyzer = GeminiImageAnalyzer()
            from app.closet.segmentation import SegmentationService

            self._segmentation = SegmentationService(
                gemini_client=self._analyzer,
            )
            self.validator = ImageValidator()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 공개 메서드
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def validate_image(self, image_url: str) -> dict:
        """단일 이미지 어뷰징 및 유효성 검증을 수행합니다."""
        return await self.validator.validate_image(image_url)

    async def preprocess(self, target_image_url: str, user_id: int) -> PreprocessResult:
        """세그멘테이션 → presigned URL 발급 → S3 업로드 (N개 아이템).

        Returns:
            PreprocessResult: N개 아이템의 file_ids, task_ids, item_images
        """
        # 1. 세그멘테이션: 모델 이미지 → N개 개별 아이템 bytes
        item_images = await self._segment(target_image_url)
        if not item_images:
            return PreprocessResult(success=False, error="SEGMENTATION_FAILED")

        count = len(item_images)
        task_ids = [str(ULID()) for _ in range(count)]

        # 2. Presigned URL N개 한 번에 발급
        presigned_list = await self._request_presigned_urls(
            user_id, purpose="CLOTHES", count=count
        )
        file_ids = [p["fileId"] for p in presigned_list]

        # 3. S3 업로드 (병렬)
        upload_tasks = [
            self._safe_upload(presigned_list[i]["presignedUrl"], item_images[i])
            for i in range(count)
        ]
        upload_results = await asyncio.gather(*upload_tasks)

        # 업로드 실패 확인
        errors = [e for e in upload_results if e is not None]
        if len(errors) == count:
            return PreprocessResult(
                success=False, error=f"ALL_UPLOADS_FAILED: {errors[0]}"
            )

        return PreprocessResult(
            file_ids=file_ids,
            task_ids=task_ids,
            item_images=item_images,
            success=True,
        )

    async def analyze(self, image_bytes: bytes) -> AnalysisResult:
        """세그멘테이션된 개별 아이템 이미지를 분석합니다.

        Args:
            image_bytes: 세그멘테이션된 아이템 이미지 bytes

        Returns:
            AnalysisResult: 분석된 major/extra 속성과 성공 여부를 담은 결과.
        """
        raw_analysis = await self._safe_analyze(image_bytes)
        normalized = self._normalize_analysis(raw_analysis)

        major = MajorAttributes(**normalized["major"])
        extra = ExtraAttributes(
            meta_data=ExtraMetadata(**normalized["extra"]["meta_data"]),
            caption=normalized["extra"]["caption"],
        )

        return AnalysisResult(major=major, extra=extra, success=True)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 내부 헬퍼 (각 단계별 안전한 실행)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    async def _segment(self, image_url: str) -> list[bytes]:
        """세그멘테이션 실행. Mock/실제 모두 self._segmentation.segment() 호출."""
        try:
            return await self._segmentation.segment(image_url)
        except Exception as e:
            logger.error(f"세그멘테이션 실패: {e}")
            return []

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
        """이미지 분석. vLLM 실패 시 Gemini 폴백, 둘 다 실패 시 DEFAULT_ANALYSIS 반환."""
        try:
            result = await self._analyzer.analyze_image(image_bytes)
            return result.model_dump()
        except Exception as e:
            if self._fallback_analyzer is not None:
                logger.warning(f"vLLM 분석 실패, Gemini 폴백 시도: {e}")
                try:
                    result = await self._fallback_analyzer.analyze_image(image_bytes)
                    return result.model_dump()
                except Exception as fallback_err:
                    logger.error(f"Gemini 폴백도 실패 (기본값 사용): {fallback_err}")
                    return DEFAULT_ANALYSIS.copy()
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

    async def _request_presigned_urls(
        self, user_id: int, purpose: str, count: int
    ) -> list[dict]:
        """백엔드 내부 API를 호출하여 S3 presigned URL을 N개 발급받습니다."""
        settings = get_settings()
        url = f"{settings.backend_internal_url}/api/internal/presigned-url"

        files = [{"name": f"item_{i}.png", "type": "image/png"} for i in range(count)]

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
                    "files": files,
                },
            )
            response.raise_for_status()

        result = response.json()
        return result["data"]

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
                "category": major.get("category") or "ETC",
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
