from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient


def test_create_embedding_api_success(
    client: TestClient,
    mock_embedding_service: AsyncMock,
) -> None:
    # Given
    mock_embedding_service.upsert.return_value = True
    request_data = {
        "userId": "user123",
        "clothesId": 1,
        "imageUrl": "http://example.com/image.jpg",
        "caption": "test caption",
        "metadata": {
            "category": "상의",
            "color": ["빨강"],
            "material": ["니트"],
            "styleTags": ["캐주얼"],
            "gender": "남성",
            "season": ["겨울"],
            "formality": "캐주얼",
            "fit": "오버핏",
        },
    }

    # When
    response = client.post("/v1/closet/embedding", json=request_data)

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["clothesId"] == 1
    assert data["indexed"] is True
    mock_embedding_service.upsert.assert_called_once()


def test_delete_embedding_api_success(
    client: TestClient,
    mock_embedding_service: AsyncMock,
) -> None:
    # Given
    mock_embedding_service.delete.return_value = True
    clothes_id = 1

    # When
    response = client.delete(f"/v1/closet/{clothes_id}")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["clothesId"] == 1
    assert data["deleted"] is True
    mock_embedding_service.delete.assert_called_once_with(clothes_id)
