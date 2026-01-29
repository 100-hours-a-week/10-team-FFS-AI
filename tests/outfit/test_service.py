from unittest.mock import AsyncMock, MagicMock

import pytest

from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
    OutfitItem,
    OutfitRequest,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
)
from app.outfit.service import OutfitService


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
            SearchQuery(text="상의 검색", category_filter="상의"),
            SearchQuery(text="하의 검색", category_filter="하의"),
        ]
    )
    return builder


@pytest.fixture
def mock_repository() -> MagicMock:
    repo = MagicMock()
    repo.search_multiple = AsyncMock(
        return_value=[
            SearchResult(
                category="상의",
                candidates=[
                    ClothingCandidate(
                        clothes_id=101,
                        image_url="https://img.com/101.jpg",
                        category="상의",
                        color=["흰색"],
                        style_tags=["포멀"],
                        caption="흰색 셔츠",
                        similarity_score=0.95,
                    )
                ],
            ),
            SearchResult(
                category="하의",
                candidates=[
                    ClothingCandidate(
                        clothes_id=201,
                        image_url="https://img.com/201.jpg",
                        category="하의",
                        color=["검정"],
                        style_tags=["포멀"],
                        caption="검정 슬랙스",
                        similarity_score=0.92,
                    )
                ],
            ),
        ]
    )
    return repo


@pytest.fixture
def mock_composer() -> MagicMock:
    composer = MagicMock()
    composer.compose = AsyncMock(
        return_value=OutfitResponse(
            query_summary="면접용 포멀 코디",
            outfits=[
                Outfit(
                    outfit_id="outfit-001",
                    description="깔끔한 비즈니스 룩",
                    items=[
                        OutfitItem(
                            clothes_id=101,
                            image_url="https://img.com/101.jpg",
                            category="상의",
                            role="상의",
                        ),
                        OutfitItem(
                            clothes_id=201,
                            image_url="https://img.com/201.jpg",
                            category="하의",
                            role="하의",
                        ),
                    ],
                )
            ],
        )
    )
    return composer


@pytest.fixture
def service(
    mock_query_parser: MagicMock,
    mock_search_builder: MagicMock,
    mock_repository: MagicMock,
    mock_composer: MagicMock,
) -> OutfitService:
    return OutfitService(
        query_parser=mock_query_parser,
        search_builder=mock_search_builder,
        repository=mock_repository,
        composer=mock_composer,
    )


class TestOutfitServiceRecommend:
    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        service: OutfitService,
        mock_query_parser: MagicMock,
        mock_search_builder: MagicMock,
        mock_repository: MagicMock,
        mock_composer: MagicMock,
    ) -> None:
        request = OutfitRequest(user_id="user123", query="면접에 입을 옷 추천해줘")

        response = await service.recommend(request)

        mock_query_parser.parse.assert_awaited_once_with("면접에 입을 옷 추천해줘")
        mock_search_builder.build.assert_called_once()
        mock_repository.search_multiple.assert_awaited_once()
        mock_composer.compose.assert_awaited_once()

        assert response.query_summary == "면접용 포멀 코디"
        assert len(response.outfits) == 1
        assert len(response.outfits[0].items) == 2

    @pytest.mark.asyncio
    async def test_passes_user_id_to_repository(
        self,
        service: OutfitService,
        mock_repository: MagicMock,
    ) -> None:
        request = OutfitRequest(user_id="user456", query="오늘 뭐 입지")

        await service.recommend(request)

        call_args = mock_repository.search_multiple.call_args
        assert call_args.kwargs["user_id"] == "user456"
