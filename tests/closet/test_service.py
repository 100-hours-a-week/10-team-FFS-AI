import time
from unittest.mock import MagicMock

import pytest
from pytest_mock import MockerFixture

from app.closet.schemas import (
    AnalyzeRequest,
    BatchStatus,
    TaskStatus,
    ValidateRequest,
    ValidationErrorCode,
)
from app.closet.service import (
    get_batch_status,
    process_image_task,
    start_analyze,
    validate_images,
)

# --- Validate Tests ---


@pytest.mark.asyncio
async def test_validate_images_success(mocker: MockerFixture) -> None:
    # Given
    request = ValidateRequest(userId=1, images=["http://test.com/valid.jpg"])

    # Mock inner helper functions
    mocker.patch("app.closet.service._check_format", return_value=True)
    mocker.patch("app.closet.service._get_file_size", return_value=1024)
    mocker.patch("app.closet.service._check_quality", return_value=True)

    # Mock AI Call
    mock_ai_results = [
        {
            "url": "http://test.com/valid.jpg",
            "nsfw": {"is_nsfw": False},
            "fashion": {"is_fashion": True},
        }
    ]
    mocker.patch(
        "app.closet.service.call_ai_model_server", return_value=mock_ai_results
    )

    # When
    response = await validate_images(request)

    # Then
    assert response.success is True
    assert response.validation_summary.passed == 1
    assert response.validation_results[0].passed is True


@pytest.mark.asyncio
async def test_validate_images_nsfw_detected(mocker: MockerFixture) -> None:
    # Given
    request = ValidateRequest(userId=1, images=["http://test.com/nsfw.jpg"])

    mocker.patch("app.closet.service._check_format", return_value=True)
    mocker.patch("app.closet.service._get_file_size", return_value=1024)

    # Mock AI Call returns NSFW
    mock_ai_results = [
        {
            "url": "http://test.com/nsfw.jpg",
            "nsfw": {"is_nsfw": True},
            "fashion": {"is_fashion": True},
        }
    ]
    mocker.patch(
        "app.closet.service.call_ai_model_server", return_value=mock_ai_results
    )

    # When
    response = await validate_images(request)

    # Then
    assert response.success is True
    assert response.validation_summary.failed == 1
    assert response.validation_results[0].passed is False
    assert response.validation_results[0].error == ValidationErrorCode.NSFW


# --- Analyze Tests ---


@pytest.mark.asyncio
async def test_start_analyze_success(
    mocker: MockerFixture, mock_redis_client: MagicMock
) -> None:
    # Given
    batch_id = "test_batch"
    request = AnalyzeRequest(
        userId=1,
        batchId=batch_id,
        images=[
            {
                "sequence": 1,
                "targetImage": "http://img.jpg",
                "taskId": "task_1",
                "fileUploadInfo": {"fileId": 1, "objectKey": "k", "presignedUrl": "u"},
            }
        ],
    )
    mock_bg_tasks = MagicMock()

    # Redis가 비어있다고 가정 (exists_batch=False)
    # conftest.py의 mock implementation이 자동으로 처리함

    # When
    response = await start_analyze(request, mock_bg_tasks)

    # Then
    assert response.batch_id == batch_id
    assert response.status == BatchStatus.IN_PROGRESS

    # Redis 호출 확인
    mock_redis_client.set_batch.assert_called_once()
    mock_redis_client.set_task.assert_called_once()
    mock_redis_client.add_task_to_batch.assert_called_once()

    mock_bg_tasks.add_task.assert_called_once()


def test_get_batch_status_completed(mock_redis_client: MagicMock) -> None:
    # Given
    batch_id = "batch_done"
    task_id = "task_1"

    # Redis Mock에 데이터 주입 (conftest.py의 side_effect 이용)
    mock_redis_client.set_batch(
        batch_id,
        {
            "user_id": "test",
            "status": BatchStatus.ACCEPTED,
            "total": 1,
            "completed": 0,
            "processing": 1,
            "created_at": time.time(),
        },
    )
    mock_redis_client.set_task(
        task_id,
        {
            "batch_id": batch_id,
            "status": TaskStatus.COMPLETED,
            "sequence": 1,
            "analysis_result": {"category": "TOP"},
        },
    )
    mock_redis_client.add_task_to_batch(batch_id, task_id)

    # When
    result = get_batch_status(batch_id)

    # Then
    assert result is not None
    # 로직상 completed count가 업데이트되어야 COMPLETED가 됨 (process_image_task에서 됨)
    # 여기선 단순히 조회 로직만 테스트하므로, 수동으로 데이터를 완벽하게 맞춰주거나
    # get_batch_status 로직이 카운트를 세는지 확인해야 함.
    # get_batch_status 구현: task loop 돌면서 completed_count 셈.
    # 따라서 위에 set_task status=COMPLETED 면 completed_count=1이 됨.
    # total=1, completed=1 => COMPLETED

    assert result.status == BatchStatus.COMPLETED
    assert result.meta.is_finished is True
    assert result.results[0].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_process_image_task_success(
    mocker: MockerFixture, mock_redis_client: MagicMock
) -> None:
    # Given
    task_id = "task_1"
    batch_id = "batch_1"

    # Redis Mock 상태 설정
    mock_redis_client.set_batch(batch_id, {"completed": 0, "processing": 1, "total": 1})
    mock_redis_client.set_task(
        task_id,
        {
            "batch_id": batch_id,
            "user_id": 1,
            "sequence": 0,
            "target_image": "http://img.jpg",
            "file_info": {"presigned_url": "http://presigned.url", "file_id": 100},
            "status": TaskStatus.PREPROCESSING,
            "created_at": time.time(),
        },
    )

    # Mocks
    mock_download = mocker.patch("app.closet.service.download_image")
    mock_download.return_value = MagicMock()  # PIL Image mock

    mock_remover = MagicMock()
    mock_remover.remove_background.return_value = MagicMock()  # Segmented Image mock
    mocker.patch("app.closet.service.get_background_remover", return_value=mock_remover)

    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "https://s3.url/segmented.png"
    mocker.patch("app.closet.service.get_storage", return_value=mock_storage)

    # analyze_image_attributes mock
    mocker.patch(
        "app.closet.service._analyze_image_attributes", return_value={"category": "TOP"}
    )

    # When
    await process_image_task(task_id)

    # Then
    # Redis 상태 확인
    updated_task = mock_redis_client.get_task(task_id)
    assert updated_task["status"] == TaskStatus.COMPLETED
    assert updated_task["file_id"] == 100

    # Check calls
    mock_download.assert_called_once_with("http://presigned.url")
    mock_remover.remove_background.assert_called_once()
    mock_storage.upload_file.assert_called_once()
