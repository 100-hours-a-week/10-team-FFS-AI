from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import BackgroundTasks

from app.closet.schemas import (
    AnalyzeImageItem,
    AnalyzeRequest,
    AnalyzeResponse,
    BatchMeta,
    BatchStatus,
    FileUploadInfo,
    TaskResult,
    TaskStatus,
)
from app.closet.service import ClosetService


@pytest.fixture
def mock_redis() -> AsyncMock:
    redis = AsyncMock()
    redis.get.return_value = None
    return redis


@pytest.fixture
def mock_s3_client() -> MagicMock:
    client = MagicMock()
    client.get_image = AsyncMock(return_value=b"fake_image_bytes")
    client.put_image = AsyncMock()
    return client


@pytest.fixture
def mock_gemini_analyzer() -> MagicMock:
    analyzer = MagicMock()
    analyzer.analyze_image = AsyncMock(
        return_value={
            "major": {
                "category": "셔츠",
                "color": ["흰색"],
                "material": [],
                "style_tags": [],
            },
            "extra": {"meta_data": {}, "caption": "흰색 셔츠입니다."},
        }
    )
    return analyzer


@pytest.fixture
def service(
    mock_redis: AsyncMock,
    mock_s3_client: MagicMock,
    mock_gemini_analyzer: MagicMock,
) -> Generator[ClosetService, Any, None]:
    with (
        patch("app.closet.service.S3Client", return_value=mock_s3_client),
        patch(
            "app.closet.service.GeminiImageAnalyzer", return_value=mock_gemini_analyzer
        ),
    ):
        svc = ClosetService(redis_client=mock_redis)
        yield svc


@pytest.mark.asyncio
async def test_start_analysis(service: ClosetService, mock_redis: AsyncMock) -> None:
    # Given
    request = AnalyzeRequest(
        user_id=1,
        batch_id="batch-001",
        images=[
            AnalyzeImageItem(
                sequence=1,
                target_image="http://example.com/img.jpg",
                task_id="task-001",
                file_upload_info=FileUploadInfo(
                    file_id=101,
                    object_key="key/img.jpg",
                    presigned_url="http://s3.com/put",
                ),
            )
        ],
    )
    background_tasks = BackgroundTasks()

    # When
    response = await service.start_analysis(request, background_tasks)

    # Then
    assert response.batch_id == "batch-001"
    assert response.status == BatchStatus.IN_PROGRESS
    mock_redis.set.assert_called_once()
    assert len(background_tasks.tasks) == 1


@pytest.mark.asyncio
async def test_process_batch_success(
    service: ClosetService, mock_redis: AsyncMock
) -> None:
    # Given
    batch_id = "batch-001"
    images = [
        AnalyzeImageItem(
            sequence=1,
            target_image="http://example.com/img.jpg",
            task_id="task-001",
            file_upload_info=FileUploadInfo(
                file_id=101, object_key="key", presigned_url="url"
            ),
        )
    ]

    # Redis get이 초기 배치 상태를 반환하도록 설정
    initial_state = AnalyzeResponse(
        batch_id=batch_id,
        status=BatchStatus.IN_PROGRESS,
        meta=BatchMeta(total=1, completed=0, processing=1),
        results=[
            TaskResult(task_id="task-001", status=TaskStatus.PREPROCESSING, file_id=101)
        ],
    )
    mock_redis.get.return_value = initial_state.model_dump_json()

    # When
    await service.process_batch(batch_id, images)

    # Then
    # 상태 업데이트 호출 확인:
    # 1. PREPROCESSING 상태 업데이트
    # 2. ANALYZING 상태 업데이트
    # 3. progress 업데이트
    # 4. finalize 업데이트
    assert mock_redis.set.call_count >= 2

    # 마지막 set 호출(완료 상태) 검증
    call_args = mock_redis.set.call_args_list[-1]
    saved_json = call_args[0][1]
    assert '"status":"COMPLETED"' in saved_json or '"status": "COMPLETED"' in saved_json


@pytest.mark.asyncio
async def test_process_batch_download_failure(
    service: ClosetService, mock_redis: AsyncMock, mock_s3_client: MagicMock
) -> None:
    # Given
    batch_id = "batch-001"
    images = [
        AnalyzeImageItem(
            sequence=1,
            target_image="http://example.com/img.jpg",
            task_id="task-001",
            file_upload_info=FileUploadInfo(
                file_id=101, object_key="key", presigned_url="url"
            ),
        )
    ]

    # 다운로드 실패 설정
    mock_s3_client.get_image.side_effect = Exception("Download failed")

    # Redis 초기 상태
    initial_state = AnalyzeResponse(
        batch_id=batch_id,
        status=BatchStatus.IN_PROGRESS,
        meta=BatchMeta(total=1, completed=0, processing=1),
        results=[
            TaskResult(task_id="task-001", status=TaskStatus.PREPROCESSING, file_id=101)
        ],
    )
    mock_redis.get.return_value = initial_state.model_dump_json()

    # When
    await service.process_batch(batch_id, images)

    # Then - 다운로드 실패 시 FAILED 상태
    call_args = mock_redis.set.call_args_list[-1]
    saved_json = call_args[0][1]
    assert (
        '"status":"PARTIAL_FAILURE"' in saved_json
        or '"status": "PARTIAL_FAILURE"' in saved_json
    )


@pytest.mark.asyncio
async def test_process_batch_analysis_failure_fallback(
    service: ClosetService, mock_redis: AsyncMock, mock_gemini_analyzer: MagicMock
) -> None:
    # Given
    batch_id = "batch-001"
    images = [
        AnalyzeImageItem(
            sequence=1,
            target_image="http://example.com/img.jpg",
            task_id="task-001",
            file_upload_info=FileUploadInfo(
                file_id=101, object_key="key", presigned_url="url"
            ),
        )
    ]

    # 분석 실패 설정
    mock_gemini_analyzer.analyze_image.side_effect = Exception("Gemini API error")

    # Redis 초기 상태
    initial_state = AnalyzeResponse(
        batch_id=batch_id,
        status=BatchStatus.IN_PROGRESS,
        meta=BatchMeta(total=1, completed=0, processing=1),
        results=[
            TaskResult(task_id="task-001", status=TaskStatus.PREPROCESSING, file_id=101)
        ],
    )
    mock_redis.get.return_value = initial_state.model_dump_json()

    # When
    await service.process_batch(batch_id, images)

    # Then - 분석 실패해도 COMPLETED (fallback)
    call_args = mock_redis.set.call_args_list[-1]
    saved_json = call_args[0][1]
    # 배치는 COMPLETED (fallback으로 처리됨)
    assert '"status":"COMPLETED"' in saved_json or '"status": "COMPLETED"' in saved_json
    # error_message에 PARTIAL 기록
    assert "PARTIAL" in saved_json or "ANALYSIS_FAILED" in saved_json


@pytest.mark.asyncio
async def test_get_batch_status_found(
    service: ClosetService, mock_redis: AsyncMock
) -> None:
    # Given
    mock_redis.get.return_value = '{"batch_id": "b1", "status": "COMPLETED", "meta": {"total": 1, "completed": 1, "processing": 0}, "results": []}'

    # When
    response = await service.get_batch_status("b1")

    # Then
    assert response is not None
    assert response.batch_id == "b1"
    assert response.status == BatchStatus.COMPLETED


@pytest.mark.asyncio
async def test_get_batch_status_not_found(
    service: ClosetService, mock_redis: AsyncMock
) -> None:
    # Given
    mock_redis.get.return_value = None

    # When
    response = await service.get_batch_status("unknown")

    # Then
    assert response is None
