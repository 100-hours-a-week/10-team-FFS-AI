import logging

from app.shop.schemas import ShopParsedQuery, ShopSearchQuery

logger = logging.getLogger(__name__)


DEFAULT_SHOP_CATEGORIES = ["TOP", "BOTTOM", "SHOES"]


class ShopSearchQueryBuilder:
    def build(self, parsed: ShopParsedQuery) -> list[ShopSearchQuery]:
        if parsed.is_full_outfit_request():
            return self._build_full_outfit_queries(parsed)
        else:
            return [self._build_single_query(parsed, parsed.target_category)]

    def _build_full_outfit_queries(
        self, parsed: ShopParsedQuery
    ) -> list[ShopSearchQuery]:
        return [
            self._build_single_query(parsed, category)
            for category in DEFAULT_SHOP_CATEGORIES
        ]

    def _build_single_query(
        self, parsed: ShopParsedQuery, category: str | None
    ) -> ShopSearchQuery:
        parts: list[str] = []

        if category:
            parts.append(category)

        if parsed.style and parsed.style != "깔끔한":
            parts.append(f"{parsed.style} 스타일")

        if parsed.season:
            parts.append(f"{parsed.season}용")

        if parsed.occasion and parsed.occasion != "일상":
            parts.append(f"{parsed.occasion}에 적합")

        for constraint in parsed.constraints[:2]:
            parts.append(constraint)

        text = ". ".join(parts)
        logger.debug(f"Built shop search query: {text} (category: {category})")

        return ShopSearchQuery(text=text, category_filter=category)
