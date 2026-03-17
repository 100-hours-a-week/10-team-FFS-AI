"""커머스 크롤러 테스트.

네이버 API 호출을 mock하여 크롤링 로직을 검증한다.
"""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.commerce.crawler import (
    NaverShoppingCrawler,
    _clean_naver_item,
    _strip_html,
)

# ── _strip_html 테스트 ──


def test_strip_html_bold() -> None:
    assert _strip_html("<b>나이키</b> 에어포스") == "나이키 에어포스"


def test_strip_html_multiple_tags() -> None:
    assert _strip_html("<b>니트</b> <em>여성</em>") == "니트 여성"


def test_strip_html_no_tags() -> None:
    assert _strip_html("태그 없음") == "태그 없음"


def test_strip_html_empty() -> None:
    assert _strip_html("") == ""


# ── _clean_naver_item 테스트 ──


def test_clean_naver_item_basic() -> None:
    """네이버 API 응답 dict를 NaverProduct로 변환."""
    item = {
        "productId": "12345",
        "title": "<b>남성</b> 반팔 티셔츠",
        "link": "https://example.com",
        "image": "https://img.example.com/img.jpg",
        "lprice": "25900",
        "mallName": "무신사",
        "brand": "나이키",
        "maker": "Nike",
        "category1": "패션의류",
        "category2": "남성의류",
        "category3": "티셔츠",
        "category4": "",
    }
    product = _clean_naver_item(item, "남성 반팔 티셔츠")

    assert product.product_id == "12345"
    assert product.title == "남성 반팔 티셔츠"  # HTML 태그 제거됨
    assert product.price == 25900
    assert product.brand == "나이키"
    assert product.category4 is None  # 빈 문자열 → None
    assert product.search_keyword == "남성 반팔 티셔츠"


def test_clean_naver_item_missing_fields() -> None:
    """필드가 누락된 경우 기본값 처리."""
    item = {"productId": "999"}
    product = _clean_naver_item(item, "test")

    assert product.product_id == "999"
    assert product.title == ""
    assert product.price == 0
    assert product.brand is None
    assert product.maker is None


def test_clean_naver_item_empty_brand_to_none() -> None:
    """brand/maker가 빈 문자열이면 None."""
    item = {"productId": "888", "brand": "", "maker": ""}
    product = _clean_naver_item(item, "test")
    assert product.brand is None
    assert product.maker is None


# ── NaverShoppingCrawler 테스트 ──


@pytest.fixture
def mock_settings() -> Generator[MagicMock, Any, None]:
    with patch("app.commerce.crawler.get_settings") as mock:
        settings = MagicMock()
        settings.naver_client_id = "test_id"
        settings.naver_client_secret = "test_secret"
        settings.batch_max_products_per_keyword = 100
        mock.return_value = settings
        yield settings


@pytest.mark.asyncio
async def test_crawl_keyword_success(mock_settings: MagicMock) -> None:
    """키워드 1개 크롤링 성공."""
    api_response = {
        "items": [
            {
                "productId": "111",
                "title": "<b>테스트</b> 상품1",
                "link": "https://a.com",
                "image": "https://img.com/1.jpg",
                "lprice": "10000",
                "mallName": "몰1",
                "brand": "브랜드A",
                "maker": "",
                "category1": "패션의류",
                "category2": "남성의류",
                "category3": "티셔츠",
                "category4": "",
            },
            {
                "productId": "222",
                "title": "상품2",
                "link": "https://b.com",
                "image": "https://img.com/2.jpg",
                "lprice": "20000",
                "mallName": "몰2",
                "brand": "",
                "maker": "",
                "category1": "패션의류",
                "category2": "",
                "category3": "",
                "category4": "",
            },
        ]
    }

    mock_response = MagicMock()
    mock_response.json.return_value = api_response
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    crawler = NaverShoppingCrawler()
    products = await crawler.crawl_keyword(mock_client, "테스트", max_products=10)

    assert len(products) == 2
    assert products[0].product_id == "111"
    assert products[0].title == "테스트 상품1"  # HTML 태그 제거
    assert products[1].brand is None  # 빈 문자열 → None


@pytest.mark.asyncio
async def test_crawl_keyword_empty_response(mock_settings: MagicMock) -> None:
    """API가 빈 결과를 반환하면 빈 리스트."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"items": []}
    mock_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    crawler = NaverShoppingCrawler()
    products = await crawler.crawl_keyword(mock_client, "없는키워드", max_products=10)

    assert products == []


@pytest.mark.asyncio
async def test_crawl_keyword_api_error(mock_settings: MagicMock) -> None:
    """API 호출 실패 시 빈 리스트 반환 (에러는 내부에서 로깅)."""
    mock_response = MagicMock()
    mock_response.status_code = 429
    mock_response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "error", request=MagicMock(), response=mock_response
        )
    )
    mock_response.json.return_value = {"items": []}

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)

    crawler = NaverShoppingCrawler()
    products = await crawler.crawl_keyword(mock_client, "테스트", max_products=10)
    assert products == []
