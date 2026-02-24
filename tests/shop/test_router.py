from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from app.shop.schemas import (
    ShopOutfit,
    ShopProduct,
    ShopSearchResponse,
)


def test_shop_search_success(
    client: TestClient,
    mock_shop_service: AsyncMock,
) -> None:
    # Given
    mock_response = ShopSearchResponse(
        query_summary="Y2K 크롭탑 코디입니다",
        outfits=[
            ShopOutfit(
                outfit_id="outfit_s001",
                items=[
                    ShopProduct(
                        product_id="prod_001",
                        title="Y2K 크롭탑",
                        brand="무신사",
                        price=29000,
                        image_url="https://img.com/1.jpg",
                        link="https://musinsa.com/1",
                        source="musinsa",
                        category="TOP",
                    ),
                ],
            ),
        ],
    )
    mock_shop_service.search.return_value = mock_response

    request_data = {
        "userId": 123,
        "query": "3만원 이하 Y2K 크롭탑 코디",
    }

    # When
    response = client.post(
        "/ai/v2/shop/outfit", json=request_data
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["querySummary"] == "Y2K 크롭탑 코디입니다"
    assert len(data["outfits"]) == 1
    assert data["outfits"][0]["outfitId"] == "outfit_s001"

    items = data["outfits"][0]["items"]
    assert len(items) == 1
    assert items[0]["productId"] == "prod_001"
    assert items[0]["price"] == 29000
    assert items[0]["brand"] == "무신사"
    mock_shop_service.search.assert_called_once()


def test_shop_search_empty_result(
    client: TestClient,
    mock_shop_service: AsyncMock,
) -> None:
    # Given
    mock_response = ShopSearchResponse(
        query_summary="검색 결과가 없습니다",
        outfits=[],
    )
    mock_shop_service.search.return_value = mock_response

    request_data = {
        "userId": 123,
        "query": "존재하지 않는 옷",
    }

    # When
    response = client.post(
        "/ai/v2/shop/outfit", json=request_data
    )

    # Then
    assert response.status_code == 200
    data = response.json()
    assert data["outfits"] == []


def test_shop_search_parse_error(
    client: TestClient,
    mock_shop_service: AsyncMock,
) -> None:
    # Given
    from app.shop.exceptions import ShopParseError

    mock_shop_service.search.side_effect = ShopParseError(
        "쿼리 파싱 실패"
    )

    request_data = {
        "userId": 123,
        "query": "",
    }

    # When
    response = client.post(
        "/ai/v2/shop/outfit", json=request_data
    )

    # Then
    assert response.status_code == 400
    data = response.json()
    assert "파싱" in data["detail"]


def test_shop_search_llm_error(
    client: TestClient,
    mock_shop_service: AsyncMock,
) -> None:
    # Given
    from app.shop.exceptions import ShopLLMError

    mock_shop_service.search.side_effect = ShopLLMError(
        "OpenAI API 오류"
    )

    request_data = {
        "userId": 123,
        "query": "코디 추천",
    }

    # When
    response = client.post(
        "/ai/v2/shop/outfit", json=request_data
    )

    # Then
    assert response.status_code == 503
    data = response.json()
    assert "AI 서비스 오류" in data["detail"]


def test_shop_search_unexpected_error(
    client: TestClient,
    mock_shop_service: AsyncMock,
) -> None:
    # Given
    mock_shop_service.search.side_effect = RuntimeError(
        "알 수 없는 오류"
    )

    request_data = {
        "userId": 123,
        "query": "코디 추천",
    }

    # When
    response = client.post(
        "/ai/v2/shop/outfit", json=request_data
    )

    # Then
    assert response.status_code == 500
    data = response.json()
    assert "쇼핑 검색" in data["detail"]
