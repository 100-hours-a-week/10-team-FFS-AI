import pytest

from app.outfit.schemas import ParsedQuery, ReferenceItem
from app.outfit.search_query_builder import SearchQueryBuilder


class TestSearchQueryBuilder:


    @pytest.fixture
    def builder(self):
        return SearchQueryBuilder()

    def test_full_outfit_request(self, builder):

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
            assert "포멀" in q.text
            assert "면접" in q.text

    def test_single_category_request(self, builder):

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

    def test_matching_request_with_reference(self, builder):

        # Given
        parsed = ParsedQuery(
            occasion="면접",
            style="포멀",
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
        assert "검정" in query_text
        assert "코트" in query_text
        assert "매칭" in query_text

    def test_with_season(self, builder):

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
        assert "겨울" in queries[0].text

    def test_casual_occasion_excluded(self, builder):

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
