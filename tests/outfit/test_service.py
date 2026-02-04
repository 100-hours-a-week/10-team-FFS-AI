from unittest.mock import AsyncMock, MagicMock

import pytest

from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
    OutfitRequest,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
)
from app.outfit.service import OutfitService
from app.outfit.vton_client import VTONResponse


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
                    )
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
                    )
                ],
            ),
        ]
    )
    return repo


@pytest.fixture
def mock_composer() -> MagicMock:
    from app.outfit.schemas import OutfitItem

    composer = MagicMock()
    composer.compose = AsyncMock(
        return_value=OutfitResponse(
            query_summary="면접용 포멀 코디",
            outfits=[
                Outfit(
                    outfit_id="outfit-001",
                    description="깔끔한 비즈니스 룩",
                    clothes_ids=[101, 201],
                    items=[
                        OutfitItem(
                            clothes_id=101,
                            image_url="https://img.com/101.jpg",
                            category="TOP",
                            role="상의",
                        ),
                        OutfitItem(
                            clothes_id=201,
                            image_url="https://img.com/201.jpg",
                            category="BOTTOM",
                            role="하의",
                        ),
                    ],
                )
            ],
        )
    )
    return composer


@pytest.fixture
def mock_vton_client() -> MagicMock:
    client = MagicMock()
    client.generate_outfit_image = AsyncMock(
        return_value=VTONResponse(status="completed", image_data=b"generated-image")
    )
    return client


@pytest.fixture
def mock_vton_processor() -> MagicMock:
    processor = MagicMock()
    processor.process = AsyncMock()
    return processor


@pytest.fixture
def service(
    mock_query_parser: MagicMock,
    mock_search_builder: MagicMock,
    mock_repository: MagicMock,
    mock_composer: MagicMock,
    mock_vton_processor: MagicMock,
) -> OutfitService:
    service = OutfitService(
        query_parser=mock_query_parser,
        search_builder=mock_search_builder,
        repository=mock_repository,
        composer=mock_composer,
        vton_processor=mock_vton_processor,
    )
    return service


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
        request = OutfitRequest(user_id=123, query="면접에 입을 옷 추천해줘")

        response = await service.recommend(request)

        mock_query_parser.parse.assert_awaited_once_with("면접에 입을 옷 추천해줘")
        mock_search_builder.build.assert_called_once()
        mock_repository.search_multiple.assert_awaited_once()
        mock_composer.compose.assert_awaited_once()

        assert response.query_summary == "면접용 포멀 코디"
        assert len(response.outfits) == 1
        assert response.outfits[0].clothes_ids == [101, 201]

    @pytest.mark.asyncio
    async def test_passes_user_id_to_repository(
        self,
        service: OutfitService,
        mock_repository: MagicMock,
    ) -> None:
        request = OutfitRequest(user_id=456, query="오늘 뭐 입지")

        await service.recommend(request)

        call_args = mock_repository.search_multiple.call_args
        assert call_args.kwargs["user_id"] == 456

    @pytest.mark.asyncio
    async def test_vton_processor_called(
        self,
        service: OutfitService,
        mock_vton_processor: MagicMock,
    ) -> None:
        # Arrange
        from app.outfit.schemas import UploadSlot

        slot = UploadSlot(
            file_id=777, object_key="test.jpg", presigned_url="https://s3.com"
        )
        request = OutfitRequest(user_id=123, query="추천해줘", urls=[slot])

        # Act
        await service.recommend(request)

        # Assert
        mock_vton_processor.process.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_vton_when_urls_missing(
        self,
        service: OutfitService,
        mock_vton_processor: MagicMock,
    ) -> None:
        request = OutfitRequest(user_id=123, query="추천해줘", urls=[])

        response = await service.recommend(request)

        assert response.outfits[0].vton_error == "VTON 미요청 (urls 없음)"
        mock_vton_processor.process.assert_not_called()
