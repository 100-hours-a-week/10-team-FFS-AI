from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_validate_success() -> None:
    """/v1/closet/validate 성공 케이스"""
    payload = {"userId": 123, "images": ["https://example.com/test_image.jpg"]}
    response = client.post("/v1/closet/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "validationResults" in data


def test_validate_fail_too_many_images() -> None:
    """/v1/closet/validate 실패 케이스 (이미지 개수 초과)"""
    payload = {
        "userId": 123,
        "images": [
            f"https://example.com/{i}.jpg" for i in range(11)
        ],  # 11개 (제한 10개)
    }
    response = client.post("/v1/closet/validate", json=payload)
    assert response.status_code == 422
    assert "VALIDATION_ERROR" in response.json()["errorCode"]


def test_analyze_flow() -> None:
    """/v1/closet/analyze API 흐름 테스트"""
    # 1. 분석 시작 요청
    payload = {
        "userId": 123,
        "batchId": "test_batch_001",
        "images": [
            {
                "sequence": 0,
                "targetImage": "https://example.com/test.jpg",
                "taskId": "test_task_001",
                "fileUploadInfo": {
                    "fileId": 1,
                    "objectKey": "key/1",
                    "presignedUrl": "https://s3.url",
                },
            }
        ],
    }
    response = client.post("/v1/closet/analyze", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert data["batchId"] == "test_batch_001"
    assert data["status"] == "ACCEPTED"

    # 2. 상태 조회 (Polling)
    response = client.get(f"/v1/closet/batches/{data['batchId']}")
    assert response.status_code == 200
    status_data = response.json()
    assert status_data["batchId"] == "test_batch_001"
    # 아직 처리 중일 것이므로 IN_PROGRESS
    assert status_data["status"] in ["ACCEPTED", "IN_PROGRESS", "COMPLETED"]
