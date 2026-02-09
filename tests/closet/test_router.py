from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.closet.schemas import (
    AnalyzeResponse,
    BatchMeta,
    BatchStatus,
)
from app.closet.service import get_closet_service
from app.main import app

client = TestClient(app)


@pytest.fixture
def mock_closet_service() -> MagicMock:
    service = MagicMock()
    service.start_analysis = AsyncMock()
    service.get_batch_status = AsyncMock()
    return service


@pytest.fixture
def override_dependency(mock_closet_service: MagicMock) -> Generator[None, Any, None]:
    app.dependency_overrides[get_closet_service] = lambda: mock_closet_service
    yield
    app.dependency_overrides = {}


def test_analyze_images_success(
    mock_closet_service: MagicMock, override_dependency: None
) -> None:
    # Given
    mock_response = AnalyzeResponse(
        batch_id="batch-123",
        status=BatchStatus.IN_PROGRESS,
        meta=BatchMeta(total=1, completed=0, processing=1, is_finished=False),
        results=[],
    )
    mock_closet_service.start_analysis.return_value = mock_response

    payload = {
        "userId": 1,
        "batchId": "batch-123",
        "images": [
            {
                "sequence": 1,
                "targetImage": "http://example.com/img.jpg",
                "taskId": "task-1",
                "fileUploadInfo": [
                    {
                        "fileId": 100,
                        "objectKey": "key/img.jpg",
                        "presignedUrl": "http://upload.com",
                    }
                ],
            }
        ],
    }

    # When
    response = client.post("/ai/v1/closet/analyze", json=payload)

    # Then
    assert response.status_code == 202
    data = response.json()
    assert data["batchId"] == "batch-123"
    assert data["status"] == "IN_PROGRESS"
    mock_closet_service.start_analysis.assert_called_once()


def test_get_batch_status_success(
    mock_closet_service: MagicMock, override_dependency: None
) -> None:
    # Given
    mock_response = AnalyzeResponse(
        batch_id="batch-123",
        status=BatchStatus.COMPLETED,
        meta=BatchMeta(total=1, completed=1, processing=0, is_finished=True),
        results=[],
    )
    mock_closet_service.get_batch_status.return_value = mock_response

    # When
    response = client.get("/ai/v1/closet/batches/batch-123")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["batchId"] == "batch-123"
    assert data["status"] == "COMPLETED"


def test_get_batch_status_not_found(
    mock_closet_service: MagicMock, override_dependency: None
) -> None:
    # Given
    mock_closet_service.get_batch_status.return_value = None

    # When
    response = client.get("/ai/v1/closet/batches/unknown")

    # Then
    assert response.status_code == 404
    data = response.json()
    assert data["detail"]["code"] == "BATCH_NOT_FOUND"
