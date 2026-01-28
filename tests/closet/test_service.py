import time
from collections.abc import Generator
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
    _batch_store,
    _task_store,
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


@pytest.fixture(autouse=True)
def clear_stores() -> Generator[None, None, None]:
    _batch_store.clear()
    _task_store.clear()
    yield


@pytest.mark.asyncio
async def test_start_analyze_success(mocker: MockerFixture) -> None:
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

    # When
    response = await start_analyze(request, mock_bg_tasks)

    # Then
    assert response.batch_id == batch_id
    assert response.status == BatchStatus.IN_PROGRESS
    assert batch_id in _batch_store
    mock_bg_tasks.add_task.assert_called_once()


def test_get_batch_status_completed() -> None:
    # Given
    batch_id = "batch_done"
    _batch_store[batch_id] = {
        "user_id": "test",
        "status": BatchStatus.ACCEPTED,
        "total": 1,
        "completed": 0,
        "processing": 1,
        "created_at": time.time(),
    }
    # Simulate a task
    task_id = "task_1"
    _task_store[task_id] = {
        "batch_id": batch_id,
        "status": TaskStatus.COMPLETED,
        "sequence": 1,
        "analysis_result": {"category": "TOP"},  # Mock result
    }

    # When
    result = get_batch_status(batch_id)

    # Then
    assert result is not None
    assert result.status == BatchStatus.COMPLETED
    assert result.meta.is_finished is True
    assert result.results[0].status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_process_image_task_success(mocker: MockerFixture) -> None:
    # Given
    task_id = "task_1"
    batch_id = "batch_1"

    # Store에 mock task 미리 저장
    _task_store[task_id] = {
        "batch_id": batch_id,
        "user_id": 1,
        "sequence": 0,
        "target_image": "http://img.jpg",
        "file_info": {"presigned_url": "http://presigned.url", "file_id": 100},
        "status": TaskStatus.PREPROCESSING,
        "created_at": time.time(),
    }
    _batch_store[batch_id] = {"completed": 0, "processing": 1}

    # Mocks
    mock_download = mocker.patch("app.closet.service.download_image")
    mock_download.return_value = MagicMock()  # PIL Image mock

    mock_remover = MagicMock()
    mock_remover.remove_background.return_value = MagicMock()  # Segmented Image mock
    mocker.patch("app.closet.service.get_background_remover", return_value=mock_remover)

    mock_storage = MagicMock()
    mock_storage.upload_file.return_value = "https://s3.url/segmented.png"
    mocker.patch("app.closet.service.get_storage", return_value=mock_storage)

    # analyze_image_attributes mock (to prevent it from running)
    mocker.patch(
        "app.closet.service._analyze_image_attributes", return_value={"category": "TOP"}
    )

    # When
    await process_image_task(task_id)

    # Then
    task = _task_store[task_id]
    assert task["status"] == TaskStatus.COMPLETED
    assert task["file_id"] == 100  # Returns input file_id

    # Check calls
    mock_download.assert_called_once_with("http://presigned.url")
    mock_remover.remove_background.assert_called_once()
    mock_storage.upload_file.assert_called_once()
