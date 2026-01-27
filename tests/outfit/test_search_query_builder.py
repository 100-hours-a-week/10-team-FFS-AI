import pytest

from app.outfit.schemas import ParsedQuery, ReferenceItem
from app.outfit.search_query_builder import SearchQueryBuilder


class TestSearchQueryBuilder:
    @pytest.fixture
    def builder(self) -> SearchQueryBuilder:
        return SearchQueryBuilder()

    def test_full_outfit_request(self, builder: SearchQueryBuilder) -> None:
        # Given
        parsed = ParsedQuery(
            occasion="면접",
            style="포멀",
            formality="포멀",
            target_category=None,
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert len(queries) == 3
        categories = [q.category_filter for q in queries]
        assert "상의" in categories
        assert "하의" in categories
        assert "아우터" in categories

        for q in queries:
            assert "포멀 스타일" in q.text
            assert "면접에 적합" in q.text

    def test_single_category_request(self, builder: SearchQueryBuilder) -> None:
        # Given
        parsed = ParsedQuery(
            occasion="일상",
            style="캐주얼",
            target_category="바지",
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert len(queries) == 1
        assert queries[0].category_filter == "바지"
        assert "바지" in queries[0].text

    def test_reference_item_not_included_in_search_query(
        self, builder: SearchQueryBuilder
    ) -> None:
        """reference_item은 검색 쿼리에 포함되지 않고 OutfitComposer에서 활용"""
        # Given
        parsed = ParsedQuery(
            occasion="면접",
            style="포멀",
            formality="포멀",
            reference_item=ReferenceItem(
                category="코트",
                color="검정",
                style="오버핏",
            ),
            target_category="바지",
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert len(queries) == 1
        query_text = queries[0].text
        assert "바지" in query_text
        assert "포멀 스타일" in query_text
        # reference_item 관련 내용은 검색 쿼리에 포함되지 않음
        assert "검정" not in query_text
        assert "코트" not in query_text
        assert "매칭" not in query_text

    def test_with_season(self, builder: SearchQueryBuilder) -> None:
        # Given
        parsed = ParsedQuery(
            occasion="일상",
            style="깔끔한",
            season="겨울",
            target_category="아우터",
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert "겨울용" in queries[0].text

    def test_casual_occasion_excluded(self, builder: SearchQueryBuilder) -> None:
        # Given
        parsed = ParsedQuery(
            occasion="일상",
            style="캐주얼",
            target_category="상의",
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert "일상" not in queries[0].text

    def test_minimal_query_with_all_null(self, builder: SearchQueryBuilder) -> None:
        """모든 필드가 null/기본값인 경우 카테고리만 포함"""
        # Given
        parsed = ParsedQuery(
            occasion="일상",
            style="깔끔한",
            target_category="상의",
        )

        # When
        queries = builder.build(parsed)

        # Then
        assert len(queries) == 1
        assert queries[0].text == "상의"
        assert queries[0].category_filter == "상의"

    def test_query_format_matches_hybrid_formatter(
        self, builder: SearchQueryBuilder
    ) -> None:
        """검색 쿼리 포맷이 HybridFormatter와 일치하는지 확인"""
        # Given
        parsed = ParsedQuery(
            occasion="면접",
            formality="포멀",
            season="겨울",
            target_category="상의",
        )

        # When
        queries = builder.build(parsed)

        # Then
        # HybridFormatter 포맷: "{category}. {formality} 스타일. {season}용. {occasion}에 적합"
        expected_parts = ["상의", "포멀 스타일", "겨울용", "면접에 적합"]
        for part in expected_parts:
            assert part in queries[0].text
