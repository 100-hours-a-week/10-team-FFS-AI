from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.outfit.schemas import Outfit, OutfitItem, OutfitResponse


def test_recommend_outfit_success(
    client: TestClient,
    mock_outfit_service: AsyncMock,
) -> None:
    # Given
    mock_response = OutfitResponse(
        query_summary="면접에 어울리는 세미포멀 코디입니다",
        outfits=[
            Outfit(
                outfit_id="outfit_001",
                description="네이비 블레이저와 슬랙스로 구성한 코디",
                items=[
                    OutfitItem(
                        clothes_id=1,
                        image_url="https://example.com/1.jpg",
                        category="상의",
                        role="상의",
                    ),
                    OutfitItem(
                        clothes_id=2,
                        image_url="https://example.com/2.jpg",
                        category="하의",
                        role="하의",
                    ),
                ],
            ),
        ],
    )
    mock_outfit_service.recommend.return_value = mock_response

    request_data = {
        "userId": "user123",
        "query": "내일 면접인데 깔끔한 코디 추천해줘",
    }

    # When
    response = client.post("/ai/v1/closet/outfit", json=request_data)

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["querySummary"] == "면접에 어울리는 세미포멀 코디입니다"
    assert len(data["outfits"]) == 1
    assert data["outfits"][0]["outfitId"] == "outfit_001"
    mock_outfit_service.recommend.assert_called_once()


def test_recommend_outfit_llm_error(
    client: TestClient,
    mock_outfit_service: AsyncMock,
) -> None:
    # Given
    from app.outfit.exceptions import LLMError

    mock_outfit_service.recommend.side_effect = LLMError("OpenAI API 오류")

    request_data = {
        "userId": "user123",
        "query": "오늘 데이트 코디 추천해줘",
    }

    # When
    response = client.post("/ai/v1/closet/outfit", json=request_data)

    # Then
    assert response.status_code == 503
    data = response.json()
    assert "AI 서비스 오류" in data["detail"]


def test_recommend_outfit_parse_error(
    client: TestClient,
    mock_outfit_service: AsyncMock,
) -> None:
    # Given
    from app.outfit.exceptions import ParseError

    mock_outfit_service.recommend.side_effect = ParseError("쿼리 파싱 실패")

    request_data = {
        "userId": "user123",
        "query": "",
    }

    # When
    response = client.post("/ai/v1/closet/outfit", json=request_data)

    # Then
    assert response.status_code == 400
    data = response.json()
    assert "파싱" in data["detail"]
