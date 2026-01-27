import logging

from app.outfit.schemas import ParsedQuery, SearchQuery

logger = logging.getLogger(__name__)

# 전체 코디 요청 시 검색할 카테고리
DEFAULT_CATEGORIES = ["상의", "하의", "아우터"]


class SearchQueryBuilder:
    """ParsedQuery를 벡터 검색용 쿼리로 변환"""

    def build(self, parsed: ParsedQuery) -> list[SearchQuery]:
        """검색 쿼리 목록 생성"""
        if parsed.is_full_outfit_request():
            return self._build_full_outfit_queries(parsed)
        else:
            return [self._build_single_query(parsed, parsed.target_category)]

    def _build_full_outfit_queries(self, parsed: ParsedQuery) -> list[SearchQuery]:
        """전체 코디용 검색 쿼리 (카테고리별)"""
        return [
            self._build_single_query(parsed, category)
            for category in DEFAULT_CATEGORIES
        ]

    def _build_single_query(self, parsed: ParsedQuery, category: str) -> SearchQuery:
        """단일 카테고리 검색 쿼리 생성"""
        parts = [category]

        # 스타일/격식 추가
        if parsed.formality:
            parts.append(f"{parsed.formality} 스타일")
        elif parsed.style:
            parts.append(f"{parsed.style} 스타일")

        # TPO 컨텍스트
        if parsed.occasion and parsed.occasion != "일상":
            parts.append(f"{parsed.occasion}에 적합")

        # 계절
        if parsed.season:
            parts.append(f"{parsed.season}용")

        # 기준 아이템 매칭 컨텍스트
        if parsed.reference_item:
            ref = parsed.reference_item
            ref_desc = self._describe_item(ref)
            if ref_desc:
                parts.append(f"{ref_desc}와 매칭")

        text = ". ".join(parts) + "."
        logger.debug(f"Built search query: {text} (category: {category})")

        return SearchQuery(text=text, category_filter=category)

    def _describe_item(self, item) -> str:
        """ReferenceItem을 텍스트로 변환"""
        parts = []
        if item.color:
            parts.append(item.color)
        if item.style:
            parts.append(item.style)
        if item.category:
            parts.append(item.category)
        return " ".join(parts)