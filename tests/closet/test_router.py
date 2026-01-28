from fastapi.testclient import TestClient
from pytest_mock import MockerFixture


def test_validate_api_success(client: TestClient, mocker: MockerFixture) -> None:
    # Given
    # Mock service function (validate_images)
    mock_validate_images = mocker.patch("app.closet.router.validate_images")
    mock_validate_images.return_value = {
        "success": True,
        "validationSummary": {"total": 2, "passed": 2, "failed": 0},
        "validationResults": [],
    }

    payload = {
        "userId": 1,
        "images": ["http://test.com/1.jpg", "http://test.com/2.jpg"],
    }

    # When
    response = client.post("/v1/closet/validate", json=payload)

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    mock_validate_images.assert_called_once()


def test_validate_api_validation_error(client: TestClient) -> None:
    # Given: Invalid payload (missing userId)
    payload = {"images": ["http://test.com/1.jpg"]}

    # When
    response = client.post("/v1/closet/validate", json=payload)

    # Then
    assert response.status_code == 400
    data = response.json()
    assert data["errorCode"] == "INVALID_REQUEST"
    assert "userId" in data["message"]


def test_analyze_api_success(client: TestClient, mocker: MockerFixture) -> None:
    # Given
    mock_start_analyze = mocker.patch("app.closet.router.start_analyze")
    mock_start_analyze.return_value = {
        "batchId": "batch_123",
        "status": "IN_PROGRESS",
        "meta": {"total": 1, "completed": 0, "processing": 1, "isFinished": False},
        "results": [],
    }

    payload = {
        "userId": 1,
        "batchId": "batch_123",
        "images": [
            {
                "sequence": 1,
                "targetImage": "http://test.com/img.jpg",
                "taskId": "task_123",
                "fileUploadInfo": {
                    "fileId": 1,
                    "objectKey": "key",
                    "presignedUrl": "http://url",
                },
            }
        ],
    }

    # When
    response = client.post("/v1/closet/analyze", json=payload)

    # Then
    assert response.status_code == 202
    data = response.json()
    assert data["batchId"] == "batch_123"
    assert data["status"] == "IN_PROGRESS"
    mock_start_analyze.assert_called_once()


def test_get_batch_status_success(client: TestClient, mocker: MockerFixture) -> None:
    # Given
    batch_id = "batch_123"
    mock_get_batch_status = mocker.patch("app.closet.router.get_batch_status")
    mock_get_batch_status.return_value = {
        "batchId": batch_id,
        "status": "COMPLETED",
        "meta": {"total": 1, "completed": 1, "processing": 0, "isFinished": True},
        "results": [],
    }

    # When
    response = client.get(f"/v1/closet/batches/{batch_id}")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["batchId"] == batch_id
    assert data["status"] == "COMPLETED"


def test_get_batch_status_not_found(client: TestClient, mocker: MockerFixture) -> None:
    # Given
    mock_get_batch_status = mocker.patch("app.closet.router.get_batch_status")
    mock_get_batch_status.return_value = None

    # When
    response = client.get("/v1/closet/batches/non_existent_id")

    # Then
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
