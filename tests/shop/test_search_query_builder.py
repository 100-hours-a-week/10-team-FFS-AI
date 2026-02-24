from app.shop.schemas import ShopParsedQuery, ShopSearchQuery
from app.shop.search_query_builder import (
    DEFAULT_SHOP_CATEGORIES,
    ShopSearchQueryBuilder,
)


class TestShopSearchQueryBuilder:
    def setup_method(self) -> None:
        self.builder = ShopSearchQueryBuilder()

    def test_full_outfit_generates_multiple_queries(
        self,
    ) -> None:
        """전체 코디 요청 시 기본 카테고리 수만큼 쿼리 생성"""
        parsed = ShopParsedQuery(style="Y2K")

        queries = self.builder.build(parsed)

        assert len(queries) == len(DEFAULT_SHOP_CATEGORIES)
        categories = [q.category_filter for q in queries]
        assert "TOP" in categories
        assert "BOTTOM" in categories
        assert "SHOES" in categories

    def test_single_category_generates_one_query(
        self,
    ) -> None:
        """특정 카테고리 요청 시 단일 쿼리 생성"""
        parsed = ShopParsedQuery(style="캐주얼", target_category="BOTTOM")

        queries = self.builder.build(parsed)

        assert len(queries) == 1
        assert queries[0].category_filter == "BOTTOM"

    def test_style_included_in_query_text(self) -> None:
        """스타일이 검색 텍스트에 포함됨"""
        parsed = ShopParsedQuery(style="Y2K", target_category="TOP")

        queries = self.builder.build(parsed)

        assert "Y2K" in queries[0].text

    def test_default_style_excluded(self) -> None:
        """기본 스타일('깔끔한')은 텍스트에 포함 안 됨"""
        parsed = ShopParsedQuery(style="깔끔한", target_category="TOP")

        queries = self.builder.build(parsed)

        assert "깔끔한" not in queries[0].text

    def test_season_included(self) -> None:
        """계절이 검색 텍스트에 포함됨"""
        parsed = ShopParsedQuery(
            style="캐주얼",
            season="여름",
            target_category="TOP",
        )

        queries = self.builder.build(parsed)

        assert "여름" in queries[0].text

    def test_occasion_included(self) -> None:
        """상황이 검색 텍스트에 포함됨 (일상 제외)"""
        parsed = ShopParsedQuery(
            style="포멀",
            occasion="면접",
            target_category="TOP",
        )

        queries = self.builder.build(parsed)

        assert "면접" in queries[0].text

    def test_default_occasion_excluded(self) -> None:
        """기본 상황('일상')은 텍스트에 포함 안 됨"""
        parsed = ShopParsedQuery(
            style="캐주얼",
            occasion="일상",
            target_category="TOP",
        )

        queries = self.builder.build(parsed)

        assert "일상" not in queries[0].text

    def test_constraints_included(self) -> None:
        """제약사항 키워드가 검색 텍스트에 포함됨 (최대 2개)"""
        parsed = ShopParsedQuery(
            style="Y2K",
            target_category="TOP",
            constraints=["크롭탑", "빈티지", "레이스"],
        )

        queries = self.builder.build(parsed)

        assert "크롭탑" in queries[0].text
        assert "빈티지" in queries[0].text
        # 3번째 제약사항은 포함되지 않음
        assert "레이스" not in queries[0].text

    def test_query_type_is_shop_search_query(self) -> None:
        """반환 타입이 ShopSearchQuery인지 확인"""
        parsed = ShopParsedQuery(style="캐주얼", target_category="TOP")

        queries = self.builder.build(parsed)

        for q in queries:
            assert isinstance(q, ShopSearchQuery)
