import logging

from app.outfit.schemas import ParsedQuery, SearchQuery

logger = logging.getLogger(__name__)

DEFAULT_CATEGORIES = ["TOP", "BOTTOM", "SHOES"]


class SearchQueryBuilder:
    def build(self, parsed: ParsedQuery) -> list[SearchQuery]:
        if parsed.is_full_outfit_request():
            return self._build_full_outfit_queries(parsed)
        else:
            category = parsed.target_category
            sub_list = (parsed.sub_categories or {}).get(category) if category else None
            if sub_list:
                return [
                    self._build_single_query(parsed, category, sub_category=sub)
                    for sub in sub_list
                ]
            return [self._build_single_query(parsed, category)]

    def _build_full_outfit_queries(self, parsed: ParsedQuery) -> list[SearchQuery]:
        """전체 코디 요청 — 카테고리별 sub_categories가 있으면 다중 쿼리 생성 후 flatten."""
        queries: list[SearchQuery] = []
        for category in DEFAULT_CATEGORIES:
            sub_list = (parsed.sub_categories or {}).get(category)
            if sub_list:
                # sub_category별로 각각 쿼리 생성 → flatten (list[SearchQuery] 유지)
                for sub in sub_list:
                    queries.append(
                        self._build_single_query(parsed, category, sub_category=sub)
                    )
            else:
                queries.append(self._build_single_query(parsed, category))
        return queries

    def _build_single_query(
        self,
        parsed: ParsedQuery,
        category: str | None,
        *,
        sub_category: str | None = None,
    ) -> SearchQuery:
        parts: list[str] = []

        category_part = f"{sub_category} {category}" if sub_category else category
        parts.append(category_part)

        if parsed.formality:
            parts.append(f"{parsed.formality} 스타일")

        if parsed.season:
            parts.append(f"{parsed.season}용")

        if parsed.occasion and parsed.occasion != "일상":
            parts.append(f"{parsed.occasion}에 적합")

        text = ". ".join(parts)
        logger.debug(
            f"Built search query: {text} (category: {category}, sub: {sub_category})"
        )

        return SearchQuery(text=text, category_filter=category)
