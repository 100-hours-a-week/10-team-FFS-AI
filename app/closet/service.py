from __future__ import annotations

import logging
from typing import Annotated

from fastapi import BackgroundTasks, Depends
from redis.asyncio import Redis

from app.closet.analyzer_protocol import ImageAnalyzer
from app.closet.gemini_client import GeminiImageAnalyzer
from app.closet.mock_analyzer import MockImageAnalyzer
from app.closet.s3_client import S3Client
from app.closet.schemas import (
    AnalyzeImageItem,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchMeta,
    BatchStatus,
    ExtraAttributes,
    ExtraMetadata,
    MajorAttributes,
    TaskResult,
    TaskStatus,
)
from app.common.llm_schemas import (
    ImageAnalysisResult,
    ImageExtraAttributes,
    ImageExtraMetadata,
    ImageMajorAttributes,
)
from app.common.metrics import (
    CLOSET_PIPELINE_ERRORS,
    CLOSET_STAGE_DURATION,
    REDIS_QUERIES,
    measure_time,
)
from app.config import get_settings
from app.core.database import get_redis_client

logger = logging.getLogger(__name__)


_FALLBACK_ANALYSIS = ImageAnalysisResult(
    major=ImageMajorAttributes(category="ETC"),
    extra=ImageExtraAttributes(
        meta_data=ImageExtraMetadata(),
        caption="의류 아이템",
    ),
)


class ClosetService:
    def __init__(
        self,
        redis_client: Annotated[Redis, Depends(get_redis_client)],
        image_analyzer: ImageAnalyzer | None = None,
    ) -> None:
        self.redis = redis_client
        self.s3_client = S3Client()
        self.image_analyzer: ImageAnalyzer = image_analyzer or GeminiImageAnalyzer()

    async def start_analysis(
        self, request: AnalyzeRequest, background_tasks: BackgroundTasks
    ) -> AnalyzeResponse:
        batch_id = request.batch_id

        initial_results = [
            TaskResult(
                task_id=img.task_id,
                status=TaskStatus.PREPROCESSING,
                file_id=None,
            )
            for img in request.images
        ]

        response = AnalyzeResponse(
            batch_id=batch_id,
            status=BatchStatus.IN_PROGRESS,
            meta=BatchMeta(
                total=len(request.images),
                completed=0,
                processing=len(request.images),
                is_finished=False,
            ),
            results=initial_results,
        )

        await self._save_batch(response)
        background_tasks.add_task(self.process_batch, batch_id, request.images)

        logger.info(f"Batch started: {batch_id} ({len(request.images)} images)")
        return response

    async def get_batch_status(self, batch_id: str) -> AnalyzeResponse | None:
        data = await self.redis.get(f"closet:batch:{batch_id}")
        REDIS_QUERIES.labels(operation="get").inc()
        if not data:
            return None
        return AnalyzeResponse.model_validate_json(data)

    @measure_time(
        stage="batch_processing",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def process_batch(
        self, batch_id: str, images: list[AnalyzeImageItem]
    ) -> None:
        results: list[TaskResult] = []

        try:
            for item in images:
                result = await self._process_single_image(batch_id, item)
                results.append(result)
                await self._update_progress(batch_id, len(images), results)

            failed_count = sum(1 for r in results if r.status == TaskStatus.FAILED)
            final_status = (
                BatchStatus.COMPLETED
                if failed_count == 0
                else BatchStatus.PARTIAL_FAILURE
            )
            await self._finalize_batch(batch_id, final_status, results)

            logger.info(
                f"Batch {batch_id} finished: {len(results) - failed_count} success, {failed_count} failed"
            )

        except Exception as e:
            logger.error(f"Batch {batch_id} critical failure: {e}")
            await self._handle_critical_failure(batch_id, images, results, str(e))

    async def _process_single_image(
        self, batch_id: str, item: AnalyzeImageItem
    ) -> TaskResult:
        task_id = item.task_id
        file_id = item.file_upload_info.file_id
        fallbacks: list[str] = []

        await self._update_task_status(batch_id, task_id, TaskStatus.PREPROCESSING)

        image_bytes, error = await self._safe_download(item.target_image)
        if image_bytes is None:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.FAILED,
                file_id=file_id,
                error_message=error,
            )

        await self._update_task_status(batch_id, task_id, TaskStatus.ANALYZING)

        analysis, error = await self._safe_analyze(image_bytes)
        if error:
            fallbacks.append(error)

        error = await self._safe_upload(
            item.file_upload_info.presigned_url, image_bytes
        )
        if error:
            fallbacks.append(error)

        return self._build_result(task_id, file_id, analysis, fallbacks)

    @measure_time(
        stage="image_download",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def _safe_download(self, url: str) -> tuple[bytes | None, str | None]:
        try:
            image_bytes = await self.s3_client.get_image(url)
            return image_bytes, None
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return None, f"DOWNLOAD_FAILED: {type(e).__name__}"

    @measure_time("image_analyzer", CLOSET_STAGE_DURATION, CLOSET_PIPELINE_ERRORS)
    async def _safe_analyze(
        self, image_bytes: bytes
    ) -> tuple[ImageAnalysisResult, str | None]:
        try:
            result = await self.image_analyzer.analyze_image(image_bytes)
            return result, None
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return _FALLBACK_ANALYSIS, f"ANALYSIS_FAILED: {type(e).__name__}"

    @measure_time(
        stage="image_upload",
        metric=CLOSET_STAGE_DURATION,
        error_counter=CLOSET_PIPELINE_ERRORS,
    )
    async def _safe_upload(self, presigned_url: str, image_bytes: bytes) -> str | None:
        try:
            await self.s3_client.put_image(presigned_url, image_bytes)
            return None
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            return f"UPLOAD_FAILED: {type(e).__name__}"

    def _build_result(
        self,
        task_id: str,
        file_id: int,
        analysis: ImageAnalysisResult,
        fallbacks: list[str],
    ) -> TaskResult:
        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            file_id=file_id,
            major=MajorAttributes(
                category=analysis.major.category,
                color=analysis.major.color,
                material=analysis.major.material,
                style_tags=analysis.major.style_tags,
            ),
            extra=ExtraAttributes(
                meta_data=ExtraMetadata(
                    gender=analysis.extra.meta_data.gender,
                    season=analysis.extra.meta_data.season,
                    formality=analysis.extra.meta_data.formality,
                    fit=analysis.extra.meta_data.fit,
                    occasion=analysis.extra.meta_data.occasion,
                ),
                caption=analysis.extra.caption,
            ),
            error_message=f"PARTIAL: {', '.join(fallbacks)}" if fallbacks else None,
        )

    async def _save_batch(self, response: AnalyzeResponse) -> None:
        key = f"closet:batch:{response.batch_id}"
        await self.redis.set(key, response.model_dump_json(), ex=3600)
        REDIS_QUERIES.labels(operation="set").inc()

    async def _update_task_status(
        self, batch_id: str, task_id: str, status: TaskStatus
    ) -> None:
        try:
            current = await self.get_batch_status(batch_id)
            if not current:
                return

            for r in current.results:
                if r.task_id == task_id:
                    r.status = status
                    break

            await self._save_batch(current)
        except Exception as e:
            logger.warning(f"Task status update failed: {e}")

    async def _update_progress(
        self, batch_id: str, total: int, results: list[TaskResult]
    ) -> None:
        try:
            current = await self.get_batch_status(batch_id)
            if not current:
                return

            processed = {r.task_id: r for r in results}
            updated = [processed.get(r.task_id, r) for r in current.results]

            completed = sum(
                1
                for r in updated
                if r.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
            )

            response = AnalyzeResponse(
                batch_id=batch_id,
                status=BatchStatus.IN_PROGRESS,
                meta=BatchMeta(
                    total=total,
                    completed=completed,
                    processing=total - completed,
                    is_finished=False,
                ),
                results=updated,
            )
            await self._save_batch(response)
        except Exception as e:
            logger.warning(f"Progress update failed: {e}")

    async def _finalize_batch(
        self, batch_id: str, status: BatchStatus, results: list[TaskResult]
    ) -> None:
        completed = sum(1 for r in results if r.status == TaskStatus.COMPLETED)
        response = AnalyzeResponse(
            batch_id=batch_id,
            status=status,
            meta=BatchMeta(
                total=len(results), completed=completed, processing=0, is_finished=True
            ),
            results=results,
        )
        await self._save_batch(response)

    async def _handle_critical_failure(
        self,
        batch_id: str,
        images: list[AnalyzeImageItem],
        partial_results: list[TaskResult],
        error: str,
    ) -> None:
        try:
            processed_ids = {r.task_id for r in partial_results}
            final_results = list(partial_results)

            for img in images:
                if img.task_id not in processed_ids:
                    final_results.append(
                        TaskResult(
                            task_id=img.task_id,
                            status=TaskStatus.FAILED,
                            file_id=img.file_upload_info.file_id,
                            error_message=f"BATCH_FAILED: {error}",
                        )
                    )

            await self._finalize_batch(
                batch_id, BatchStatus.PARTIAL_FAILURE, final_results
            )
        except Exception as e:
            logger.error(f"Critical failure handler failed: {e}")


def get_closet_service(
    redis_client: Annotated[Redis, Depends(get_redis_client)],
) -> ClosetService:
    settings = get_settings()
    analyzer: ImageAnalyzer | None = None
    if settings.use_mock_analyzer:
        analyzer = MockImageAnalyzer(delay_seconds=4.0)
        logger.info("Using MockImageAnalyzer for load testing")
    return ClosetService(redis_client=redis_client, image_analyzer=analyzer)
