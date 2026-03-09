from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.builder import build_outfit_graph
from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
    OutfitItem,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
    UploadSlot,
)
from app.shop.schemas import ShopSearchQuery


@pytest.fixture
def mock_query_parser() -> MagicMock:
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
def mock_repository() -> MagicMock:
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
                        color=["흰색"],
                        style_tags=["포멀"],
                        caption="흰색 셔츠",
                        similarity_score=0.95,
                    ),
                    ClothingCandidate(
                        clothes_id=102,
                        image_url="https://img.com/102.jpg",
                        category="TOP",
                        color=["하늘색"],
                        style_tags=["포멀"],
                        caption="하늘색 셔츠",
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
                        color=["검정"],
                        style_tags=["포멀"],
                        caption="검정 슬랙스",
                        similarity_score=0.92,
                    ),
                    ClothingCandidate(
                        clothes_id=202,
                        image_url="https://img.com/202.jpg",
                        category="BOTTOM",
                        color=["네이비"],
                        style_tags=["포멀"],
                        caption="네이비 슬랙스",
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
                        color=["검정"],
                        style_tags=["포멀"],
                        caption="검정 구두",
                        similarity_score=0.85,
                    ),
                ],
            ),
        ]
    )
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
def mock_composer() -> MagicMock:
    composer = MagicMock()

    valid_combinations = [
        ([101, 201, 301], "흰색 셔츠 + 검정 슬랙스 + 검정 구두"),
        ([102, 202, 301], "하늘색 셔츠 + 네이비 슬랙스 + 검정 구두"),
        ([101, 202, 301], "흰색 셔츠 + 네이비 슬랙스 + 검정 구두"),
    ]
    composer.compose = AsyncMock(
        return_value=OutfitResponse(
            query_summary="면접용 포멀 코디",
            outfits=[
                Outfit(
                    outfit_id=f"outfit-{i:03d}",
                    description=desc,
                    clothes_ids=ids,
                    items=[
                        OutfitItem(
                            clothes_id=ids[0],
                            image_url=f"https://img.com/{ids[0]}.jpg",
                            category="TOP",
                            role="상의",
                        ),
                        OutfitItem(
                            clothes_id=ids[1],
                            image_url=f"https://img.com/{ids[1]}.jpg",
                            category="BOTTOM",
                            role="하의",
                        ),
                        OutfitItem(
                            clothes_id=ids[2],
                            image_url=f"https://img.com/{ids[2]}.jpg",
                            category="SHOES",
                            role="신발",
                        ),
                    ],
                )
                for i, (ids, desc) in enumerate(valid_combinations)
            ],
        )
    )
    return composer


@pytest.fixture
def mock_vton_processor() -> MagicMock:
    processor = MagicMock()
    processor.process = AsyncMock()
    return processor


@pytest.fixture
def graph_config(
    mock_query_parser: MagicMock,
    mock_search_builder: MagicMock,
    mock_repository: MagicMock,
    mock_composer: MagicMock,
    mock_vton_processor: MagicMock,
    mock_shop_repository: MagicMock,
    mock_shop_search_builder: MagicMock,
) -> dict:
    return {
        "configurable": {
            "query_parser": mock_query_parser,
            "search_builder": mock_search_builder,
            "clothing_repository": mock_repository,
            "outfit_composer": mock_composer,
            "vton_processor": mock_vton_processor,
            "shop_repository": mock_shop_repository,
            "shop_search_builder": mock_shop_search_builder,
        }
    }


@pytest.fixture
def compiled_graph() -> CompiledStateGraph:
    return build_outfit_graph()


class TestOutfitGraph:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_query_parser: MagicMock,
        mock_search_builder: MagicMock,
        mock_repository: MagicMock,
        mock_composer: MagicMock,
    ) -> None:
        initial_state = {
            "query": "면접에 입을 옷 추천해줘",
            "user_id": 123,
            "session_id": "sess-001",
            "trace_id": "trace-001",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        mock_query_parser.parse.assert_awaited()
        mock_search_builder.build.assert_called()
        mock_repository.search_multiple.assert_awaited()
        mock_composer.compose.assert_awaited()

        response = result["response"]
        assert response.query_summary == "면접용 포멀 코디"
        assert len(response.outfits) == 3
        assert response.session_id == "sess-001"

    @pytest.mark.asyncio
    async def test_empty_search_results(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_repository: MagicMock,
        mock_composer: MagicMock,
    ) -> None:
        mock_repository.search_multiple = AsyncMock(return_value=[])
        mock_composer.compose = AsyncMock(
            return_value=OutfitResponse(
                query_summary="검색 결과 없음",
                outfits=[],
            )
        )

        initial_state = {
            "query": "없는 옷 추천해줘",
            "user_id": 123,
            "trace_id": "trace-002",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        response = result["response"]
        assert len(response.outfits) == 0
        assert result["category_coverage"] == {}
        assert result.get("shop_supplemented") is True

    @pytest.mark.asyncio
    async def test_vton_error_when_no_urls(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_vton_processor: MagicMock,
    ) -> None:
        initial_state = {
            "query": "추천해줘",
            "user_id": 123,
            "trace_id": "trace-003",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        mock_vton_processor.process.assert_not_called()
        assert result["vton_completed"] is True
        for outfit in result["response"].outfits:
            assert outfit.vton_error == "VTON 미요청 (urls 없음)"

    @pytest.mark.asyncio
    async def test_category_coverage_calculation(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
    ) -> None:
        initial_state = {
            "query": "추천해줘",
            "user_id": 123,
            "trace_id": "trace-004",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        assert result["category_coverage"] == {"TOP": 2, "BOTTOM": 2, "SHOES": 1}

    @pytest.mark.asyncio
    async def test_session_id_propagation(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
    ) -> None:
        initial_state = {
            "query": "추천해줘",
            "user_id": 123,
            "session_id": "my-session-123",
            "trace_id": "trace-005",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        assert result["response"].session_id == "my-session-123"

    @pytest.mark.asyncio
    async def test_vton_processor_called_with_slots(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_vton_processor: MagicMock,
    ) -> None:
        slot = UploadSlot(
            file_id=777, object_key="test.jpg", presigned_url="https://s3.com"
        )
        initial_state = {
            "query": "추천해줘",
            "user_id": 123,
            "trace_id": "trace-006",
            "upload_slots": [slot],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        mock_vton_processor.process.assert_awaited_once()
        assert result["vton_completed"] is True

    @pytest.mark.asyncio
    async def test_tpo_extract_error_triggers_fallback(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_query_parser: MagicMock,
    ) -> None:
        from app.outfit.exceptions import LLMError

        mock_query_parser.parse = AsyncMock(side_effect=LLMError("API timeout"))

        initial_state = {
            "query": "추천해줘",
            "user_id": 123,
            "trace_id": "trace-err-001",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        assert result.get("tpo_fallback_used") is True
        assert result["parsed_query"].occasion == "일상"
        assert result["parsed_query"].style == "깔끔한"
        assert len(result["response"].outfits) == 3

    @pytest.mark.asyncio
    async def test_parse_error_triggers_fallback(
        self,
        compiled_graph: CompiledStateGraph,
        graph_config: dict,
        mock_query_parser: MagicMock,
    ) -> None:
        from app.outfit.exceptions import ParseError

        mock_query_parser.parse = AsyncMock(side_effect=ParseError("Parse failed"))

        initial_state = {
            "query": "이상한 쿼리",
            "user_id": 123,
            "trace_id": "trace-err-002",
            "upload_slots": [],
        }

        result = await compiled_graph.ainvoke(initial_state, config=graph_config)

        assert result.get("tpo_fallback_used") is True
        assert len(result["response"].outfits) == 3
