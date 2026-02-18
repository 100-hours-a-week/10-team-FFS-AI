from __future__ import annotations

import logging
from typing import Annotated

from fastapi import BackgroundTasks, Depends
from redis.asyncio import Redis

from app.closet.analyzer_protocol import ImageAnalyzer
from app.closet.mock_analyzer import MockImageAnalyzer
from app.closet.model_analyzer import ModelServerAnalyzer
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
from app.common.metrics import (
    CLOSET_PIPELINE_ERRORS,
    CLOSET_STAGE_DURATION,
    REDIS_QUERIES,
    measure_time,
)
from app.config import get_settings
from app.core.database import get_redis_client

logger = logging.getLogger(__name__)


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


class ClosetService:
    def __init__(
        self,
        redis_client: Annotated[Redis, Depends(get_redis_client)],
        image_analyzer: ImageAnalyzer | None = None,
    ) -> None:
        self.redis = redis_client
        self.s3_client = S3Client()
        if image_analyzer:
            self.image_analyzer: ImageAnalyzer = image_analyzer
        elif get_settings().use_mock_analyzer:
            self.image_analyzer = MockImageAnalyzer()
        else:
            self.image_analyzer = ModelServerAnalyzer()

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
    async def _safe_analyze(self, image_bytes: bytes) -> tuple[dict, str | None]:
        try:
            result = await self.image_analyzer.analyze_image(image_bytes)
            return self._normalize_analysis(result), None
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return DEFAULT_ANALYSIS.copy(), f"ANALYSIS_FAILED: {type(e).__name__}"

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

    def _normalize_analysis(self, data: dict) -> dict:
        major = data.get("major", {})
        extra = data.get("extra", {})
        meta = extra.get("meta_data", {})

        return {
            "major": {
                "category": major.get("category") or "UNKNOWN",
                "color": self._to_list(major.get("color")),
                "material": self._to_list(major.get("material")),
                "style_tags": self._to_list(major.get("style_tags")),
            },
            "extra": {
                "meta_data": {
                    "gender": meta.get("gender"),
                    "season": self._to_list(meta.get("season")),
                    "formality": meta.get("formality"),
                    "fit": meta.get("fit"),
                    "occasion": self._to_list(meta.get("occasion")),
                },
                "caption": extra.get("caption") or "의류 아이템",
            },
        }

    @staticmethod
    def _to_list(value: object | None) -> list[object]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value] if value else []

    def _build_result(
        self,
        task_id: str,
        file_id: int,
        analysis: dict,
        fallbacks: list[str],
    ) -> TaskResult:
        major = analysis["major"]
        extra = analysis["extra"]
        meta = extra["meta_data"]

        return TaskResult(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            file_id=file_id,
            major=MajorAttributes(**major),
            extra=ExtraAttributes(
                meta_data=ExtraMetadata(**meta),
                caption=extra.get("caption"),
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
