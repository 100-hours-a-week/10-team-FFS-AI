"""네이버 쇼핑 API 크롤러.

키워드 기반으로 네이버 쇼핑 상품을 수집하고 중복을 제거한다.
"""

import asyncio
import logging
import re

import httpx

from app.commerce.schemas import NaverProduct
from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# 카테고리별 검색 키워드 (총 64개)
SEARCH_KEYWORDS: dict[str, list[str]] = {
    "TOP": [
        "남성 반팔 티셔츠",
        "남성 긴팔 티셔츠",
        "남성 맨투맨",
        "남성 후드티",
        "남성 셔츠",
        "남성 니트",
        "남성 가디건",
        "남성 자켓",
        "남성 코트",
        "남성 패딩",
        "남성 바람막이",
        "여성 반팔 티셔츠",
        "여성 긴팔 티셔츠",
        "여성 블라우스",
        "여성 니트",
        "여성 가디건",
        "여성 맨투맨",
        "여성 후드티",
        "여성 자켓",
        "여성 코트",
        "여성 패딩",
        "여성 크롭탑",
    ],
    "BOTTOM": [
        "남성 청바지",
        "남성 슬랙스",
        "남성 면바지",
        "남성 반바지",
        "남성 조거팬츠",
        "남성 카고바지",
        "여성 청바지",
        "여성 슬랙스",
        "여성 와이드팬츠",
        "여성 미니스커트",
        "여성 롱스커트",
        "여성 반바지",
        "여성 레깅스",
    ],
    "DRESS": [
        "미니 원피스",
        "미디 원피스",
        "롱 원피스",
        "셔츠 원피스",
        "니트 원피스",
    ],
    "SHOES": [
        "남성 스니커즈",
        "여성 스니커즈",
        "남성 구두",
        "여성 구두",
        "남성 로퍼",
        "여성 로퍼",
        "여성 부츠",
        "남성 부츠",
        "여성 샌들",
        "남성 샌들",
        "여성 힐",
    ],
    "ACCESSORY": [
        "크로스백",
        "토트백",
        "숄더백",
        "백팩",
        "클러치백",
        "남성 모자",
        "여성 모자",
        "목걸이",
        "귀걸이",
        "시계",
        "머플러",
        "벨트",
        "선글라스",
    ],
}


def _strip_html(text: str) -> str:
    """HTML 태그 제거 (네이버 API 응답의 <b> 태그 등)."""
    return re.sub(r"<[^>]+>", "", text)


def _clean_naver_item(item: dict, keyword: str) -> NaverProduct:
    """네이버 API 응답 1개를 NaverProduct로 변환."""
    return NaverProduct(
        product_id=item.get("productId", ""),
        title=_strip_html(item.get("title", "")),
        link=item.get("link", ""),
        image_url=item.get("image", ""),
        price=int(item.get("lprice", "0")),
        mall_name=item.get("mallName", ""),
        brand=item.get("brand", "") or None,
        maker=item.get("maker", "") or None,
        category1=item.get("category1", ""),
        category2=item.get("category2", "") or None,
        category3=item.get("category3", "") or None,
        category4=item.get("category4", "") or None,
        search_keyword=keyword,
    )


class NaverShoppingCrawler:
    """네이버 쇼핑 검색 API를 통해 상품을 수집한다."""

    API_URL = "https://openapi.naver.com/v1/search/shop.json"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    async def _search(
        self,
        client: httpx.AsyncClient,
        query: str,
        display: int = 100,
        start: int = 1,
    ) -> list[NaverProduct]:
        """네이버 쇼핑 검색 API 1회 호출."""
        response = await client.get(
            self.API_URL,
            headers={
                "X-Naver-Client-Id": self.settings.naver_client_id,
                "X-Naver-Client-Secret": self.settings.naver_client_secret,
            },
            params={
                "query": query,
                "display": display,
                "start": start,
                "sort": "sim",
            },
        )
        response.raise_for_status()

        return [
            _clean_naver_item(item, query) for item in response.json().get("items", [])
        ]

    async def crawl_keyword(
        self,
        client: httpx.AsyncClient,
        keyword: str,
        max_products: int = 100,
    ) -> list[NaverProduct]:
        """키워드 1개에 대해 페이지네이션하며 상품을 수집한다.

        Args:
            client: httpx 비동기 클라이언트
            keyword: 검색 키워드
            max_products: 키워드당 최대 수집 수 (최대 1000)

        Returns:
            수집된 NaverProduct 리스트
        """
        max_products = min(max_products, 1000)
        all_items: list[NaverProduct] = []
        display = 100

        for start in range(1, max_products + 1, display):
            try:
                items = await self._search(
                    client, keyword, display=display, start=start
                )
                all_items.extend(items)

                if len(items) < display:
                    break

            except httpx.HTTPStatusError as e:
                logger.error(
                    "네이버 API 에러: keyword='%s', start=%d, status=%d",
                    keyword,
                    start,
                    e.response.status_code,
                )
                break
            except Exception as e:
                logger.error("네이버 API 호출 실패: keyword='%s', error=%s", keyword, e)
                break

            await asyncio.sleep(0.15)

        return all_items

    async def crawl_all(
        self,
        max_products_per_keyword: int = 100,
    ) -> dict[str, list[NaverProduct]]:
        """전체 키워드를 순회하며 카테고리별 상품을 수집한다.

        Returns:
            카테고리별 NaverProduct 리스트 (중복 제거 완료)
        """
        result: dict[str, list[NaverProduct]] = {}
        seen_ids: set[str] = set()
        total_keywords = sum(len(kws) for kws in SEARCH_KEYWORDS.values())
        keyword_idx = 0

        async with httpx.AsyncClient(timeout=10.0) as client:
            for category, keywords in SEARCH_KEYWORDS.items():
                category_products: list[NaverProduct] = []

                for keyword in keywords:
                    keyword_idx += 1
                    logger.info(
                        "[%d/%d] 검색: '%s' (카테고리: %s)",
                        keyword_idx,
                        total_keywords,
                        keyword,
                        category,
                    )

                    try:
                        items = await self.crawl_keyword(
                            client,
                            keyword,
                            max_products_per_keyword,
                        )

                        new_count = 0
                        for item in items:
                            if item.product_id not in seen_ids:
                                seen_ids.add(item.product_id)
                                category_products.append(item)
                                new_count += 1

                        logger.info(
                            "  -> %d개 수집, %d개 신규 (카테고리 누적: %d개)",
                            len(items),
                            new_count,
                            len(category_products),
                        )

                    except Exception as e:
                        logger.error("  -> 키워드 '%s' 크롤링 실패: %s", keyword, e)

                    await asyncio.sleep(0.15)

                result[category] = category_products
                logger.info(
                    "카테고리 '%s' 완료: %d개 고유 상품",
                    category,
                    len(category_products),
                )

        total = sum(len(v) for v in result.values())
        logger.info("전체 크롤링 완료: %d개 고유 상품", total)
        return result
