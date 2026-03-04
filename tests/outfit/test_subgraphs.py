"""Phase 3 서브그래프 독립 테스트.

각 서브그래프가 독립적으로 정상 동작하는지 검증한다.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.subgraphs import (
    build_compose_subgraph,
    build_search_subgraph,
    build_tpo_subgraph,
)
from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
    OutfitItem,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
)
from app.shop.schemas import ShopSearchQuery


# ============================================================================
# TPO 서브그래프 테스트
# ============================================================================


@pytest.fixture
def mock_query_parser_success() -> MagicMock:
    """TPO 파싱 성공 mock."""
    parser = MagicMock()
    parser.parse = AsyncMock(
        return_value=ParsedQuery(
            occasion="면접",
            style="포멀",
            season="가을",
        )
    )
    return parser


@pytest.fixture
def mock_query_parser_fail() -> MagicMock:
    """TPO 파싱 실패 mock."""
    from app.outfit.exceptions import LLMError

    parser = MagicMock()
    parser.parse = AsyncMock(side_effect=LLMError("API timeout"))
    return parser


@pytest.fixture
def mock_search_builder() -> MagicMock:
    builder = MagicMock()
    builder.build = MagicMock(
        return_value=[
            SearchQuery(text="TOP 검색", category_filter="TOP"),
            SearchQuery(text="BOTTOM 검색", category_filter="BOTTOM"),
        ]
    )
    return builder


@pytest.fixture
def tpo_subgraph() -> CompiledStateGraph:
    return build_tpo_subgraph()


class TestTPOSubgraph:
    @pytest.mark.asyncio
    async def test_tpo_success_path(
        self,
        tpo_subgraph: CompiledStateGraph,
        mock_query_parser_success: MagicMock,
        mock_search_builder: MagicMock,
    ) -> None:
        """TPO 파싱 성공 시 정상 경로로 진행한다."""
        config = {
            "configurable": {
                "query_parser": mock_query_parser_success,
                "search_builder": mock_search_builder,
            }
        }
        initial_state = {
            "query": "면접에 입을 옷 추천해줘",
            "user_id": 123,
            "trace_id": "test-001",
        }

        result = await tpo_subgraph.ainvoke(initial_state, config=config)

        assert result["parsed_query"].occasion == "면접"
        assert result["parsed_query"].style == "포멀"
        assert result["quality_passed"] is True
        assert result.get("tpo_fallback_used") is not True

    @pytest.mark.asyncio
    async def test_tpo_fallback_after_retries(
        self,
        tpo_subgraph: CompiledStateGraph,
        mock_query_parser_fail: MagicMock,
        mock_search_builder: MagicMock,
    ) -> None:
        """TPO 파싱 실패 시 재시도 후 폴백으로 진행한다."""
        config = {
            "configurable": {
                "query_parser": mock_query_parser_fail,
                "search_builder": mock_search_builder,
            }
        }
        initial_state = {
            "query": "겨울 코디 추천해줘",
            "user_id": 123,
            "trace_id": "test-002",
        }

        result = await tpo_subgraph.ainvoke(initial_state, config=config)

        assert result["tpo_fallback_used"] is True
        assert result["parsed_query"].occasion == "일상"
        assert result["parsed_query"].style == "깔끔한"
        assert result["quality_passed"] is True


# ============================================================================
# 검색+보충 서브그래프 테스트
# ============================================================================


@pytest.fixture
def mock_repository_sufficient() -> MagicMock:
    """충분한 후보를 반환하는 mock repository."""
    repo = MagicMock()
    repo.search_multiple = AsyncMock(
        return_value=[
            SearchResult(
                category="TOP",
                candidates=[
                    ClothingCandidate(
                        clothes_id=101,
                        image_url="https://img.com/101.jpg",
                        category="TOP",
                        similarity_score=0.95,
                    ),
                    ClothingCandidate(
                        clothes_id=102,
                        image_url="https://img.com/102.jpg",
                        category="TOP",
                        similarity_score=0.90,
                    ),
                ],
            ),
            SearchResult(
                category="BOTTOM",
                candidates=[
                    ClothingCandidate(
                        clothes_id=201,
                        image_url="https://img.com/201.jpg",
                        category="BOTTOM",
                        similarity_score=0.92,
                    ),
                    ClothingCandidate(
                        clothes_id=202,
                        image_url="https://img.com/202.jpg",
                        category="BOTTOM",
                        similarity_score=0.88,
                    ),
                ],
            ),
            SearchResult(
                category="SHOES",
                candidates=[
                    ClothingCandidate(
                        clothes_id=301,
                        image_url="https://img.com/301.jpg",
                        category="SHOES",
                        similarity_score=0.85,
                    ),
                ],
            ),
        ]
    )
    return repo


@pytest.fixture
def mock_repository_empty() -> MagicMock:
    """빈 결과를 반환하는 mock repository."""
    repo = MagicMock()
    repo.search_multiple = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_shop_repository() -> MagicMock:
    repo = MagicMock()
    repo.search_multiple = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def mock_shop_search_builder() -> MagicMock:
    builder = MagicMock()
    builder.build = MagicMock(
        return_value=[ShopSearchQuery(text="검색 쿼리", category_filter="SHOES")]
    )
    return builder


@pytest.fixture
def search_subgraph() -> CompiledStateGraph:
    return build_search_subgraph()


class TestSearchSubgraph:
    @pytest.mark.asyncio
    async def test_sufficient_results_path(
        self,
        search_subgraph: CompiledStateGraph,
        mock_repository_sufficient: MagicMock,
        mock_shop_repository: MagicMock,
        mock_shop_search_builder: MagicMock,
    ) -> None:
        """충분한 검색 결과가 있으면 쇼핑 보충 없이 종료한다."""
        config = {
            "configurable": {
                "clothing_repository": mock_repository_sufficient,
                "shop_repository": mock_shop_repository,
                "shop_search_builder": mock_shop_search_builder,
            }
        }
        initial_state = {
            "user_id": 123,
            "trace_id": "test-003",
            "search_queries": [
                SearchQuery(text="TOP 검색", category_filter="TOP"),
                SearchQuery(text="BOTTOM 검색", category_filter="BOTTOM"),
            ],
            "required_categories": ["TOP", "BOTTOM", "SHOES"],
        }

        result = await search_subgraph.ainvoke(initial_state, config=config)

        assert result["category_coverage"] == {"TOP": 2, "BOTTOM": 2, "SHOES": 1}
        assert len(result["merged_candidates"]) == 3
        assert result.get("shop_supplemented") is not True

    @pytest.mark.asyncio
    async def test_empty_results_fast_path(
        self,
        search_subgraph: CompiledStateGraph,
        mock_repository_empty: MagicMock,
        mock_shop_repository: MagicMock,
        mock_shop_search_builder: MagicMock,
    ) -> None:
        """전 카테고리 0건이면 재검색 스킵하고 쇼핑 보충으로 직행한다."""
        config = {
            "configurable": {
                "clothing_repository": mock_repository_empty,
                "shop_repository": mock_shop_repository,
                "shop_search_builder": mock_shop_search_builder,
            }
        }
        initial_state = {
            "user_id": 999,
            "trace_id": "test-004",
            "parsed_query": ParsedQuery(occasion="일상", style="캐주얼"),
            "search_queries": [SearchQuery(text="검색", category_filter="TOP")],
            "required_categories": ["TOP", "BOTTOM", "SHOES"],
        }

        result = await search_subgraph.ainvoke(initial_state, config=config)

        assert result["shop_supplemented"] is True
        assert result.get("search_retry_count", 0) == 0


# ============================================================================
# 조합 서브그래프 테스트
# ============================================================================


@pytest.fixture
def mock_composer_3outfits() -> MagicMock:
    """3세트 코디를 반환하는 mock composer."""
    composer = MagicMock()
    composer.compose = AsyncMock(
        return_value=OutfitResponse(
            query_summary="테스트 코디",
            outfits=[
                Outfit(
                    outfit_id=f"outfit-{i}",
                    description=f"코디 {i}",
                    clothes_ids=[100 + i, 200 + i],
                    items=[
                        OutfitItem(
                            clothes_id=100 + i,
                            image_url=f"https://img.com/{100+i}.jpg",
                            category="TOP",
                            role="상의",
                        ),
                        OutfitItem(
                            clothes_id=200 + i,
                            image_url=f"https://img.com/{200+i}.jpg",
                            category="BOTTOM",
                            role="하의",
                        ),
                    ],
                )
                for i in range(3)
            ],
        )
    )
    return composer


@pytest.fixture
def mock_composer_1outfit() -> MagicMock:
    """1세트만 반환하는 mock composer."""
    composer = MagicMock()
    composer.compose = AsyncMock(
        return_value=OutfitResponse(
            query_summary="테스트 코디",
            outfits=[
                Outfit(
                    outfit_id="outfit-1",
                    description="코디 1",
                    clothes_ids=[101, 201],
                    items=[],
                )
            ],
        )
    )
    return composer


@pytest.fixture
def compose_subgraph() -> CompiledStateGraph:
    return build_compose_subgraph()


class TestComposeSubgraph:
    @pytest.mark.asyncio
    async def test_sufficient_outfits_path(
        self,
        compose_subgraph: CompiledStateGraph,
        mock_composer_3outfits: MagicMock,
    ) -> None:
        """3세트 이상 생성되면 정상 종료한다."""
        config = {"configurable": {"outfit_composer": mock_composer_3outfits}}
        initial_state = {
            "user_id": 123,
            "trace_id": "test-005",
            "parsed_query": ParsedQuery(occasion="면접", style="포멀"),
            "merged_candidates": [],
        }

        result = await compose_subgraph.ainvoke(initial_state, config=config)

        assert len(result["outfits"]) == 3
        assert result["quality_passed"] is True
        assert result.get("fallback_used") is not True

    @pytest.mark.asyncio
    async def test_insufficient_outfits_fallback(
        self,
        compose_subgraph: CompiledStateGraph,
        mock_composer_1outfit: MagicMock,
    ) -> None:
        """3세트 미만이면 재시도 후 폴백 플래그가 설정된다."""
        config = {"configurable": {"outfit_composer": mock_composer_1outfit}}
        initial_state = {
            "user_id": 123,
            "trace_id": "test-006",
            "parsed_query": ParsedQuery(occasion="면접", style="포멀"),
            "merged_candidates": [],
        }

        result = await compose_subgraph.ainvoke(initial_state, config=config)

        assert len(result["outfits"]) == 1
        assert result["fallback_used"] is True
        assert "1개만 생성됨" in result["fallback_reason"]
