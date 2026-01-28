import logging

from app.outfit.schemas import ParsedQuery, SearchQuery

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["상의", "하의", "아우터"]


class SearchQueryBuilder:
    def build(self, parsed: ParsedQuery) -> list[SearchQuery]:
        if parsed.is_full_outfit_request():
            return self._build_full_outfit_queries(parsed)
        else:
            return [self._build_single_query(parsed, parsed.target_category)]

    def _build_full_outfit_queries(self, parsed: ParsedQuery) -> list[SearchQuery]:
        return [
            self._build_single_query(parsed, category)
            for category in DEFAULT_CATEGORIES
        ]

    def _build_single_query(self, parsed: ParsedQuery, category: str) -> SearchQuery:
        parts: list[str] = []

        parts.append(category)

        if parsed.formality:
            parts.append(f"{parsed.formality} 스타일")

        if parsed.season:
            parts.append(f"{parsed.season}용")

        if parsed.occasion and parsed.occasion != "일상":
            parts.append(f"{parsed.occasion}에 적합")

        text = ". ".join(parts)
        logger.debug(f"Built search query: {text} (category: {category})")

        return SearchQuery(text=text, category_filter=category)
