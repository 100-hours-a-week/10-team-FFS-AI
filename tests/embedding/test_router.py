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
        "userId": 123,
        "clothesId": 1,
        "imageUrl": "https://s3.example.com/asdfasdfasdfasdf.png",
        "major": {
            "category": "TOP",
            "color": ["빨강"],
            "material": ["니트"],
            "styleTags": ["캐주얼", "따뜻한"],
        },
        "extra": {
            "metaData": {
                "gender": "남녀공용",
                "season": ["봄", "가을"],
                "formality": "세미 포멀",
                "fit": "오버핏",
                "occasion": ["면접", "비즈니스 미팅", "출근"],
            },
            "caption": "골드 버튼 디테일이 들어간 캐주얼한 스타일의 빨간색 니트입니다.",
        },
    }

    # When
    response = client.post("/ai/v1/closet/embedding", json=request_data)

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
    response = client.delete(f"/ai/v1/closet/{clothes_id}")

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["clothesId"] == 1
    assert data["deleted"] is True
    mock_embedding_service.delete.assert_called_once_with(clothes_id)
