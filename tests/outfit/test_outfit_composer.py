import pytest
from unittest.mock import MagicMock

from app.common.llm_schemas import (
    OutfitCombination,
    OutfitCompositionLLMResponse,
    OutfitItemLLM,
)
from app.outfit.outfit_composer import OutfitComposer
from app.outfit.schemas import ClothingCandidate, ParsedQuery, SearchResult


@pytest.fixture
def composer() -> OutfitComposer:
    # 순수 로직 테스트에서는 llm_client가 불필요하므로 Mock 주입
    # llm_client=None 전달 시 OpenAIClient()가 즉시 생성되어 OPENAI_API_KEY 예외 발생
    return OutfitComposer(llm_client=MagicMock())


@pytest.fixture
def sample_parsed_query() -> ParsedQuery:
    return ParsedQuery(
        occasion="면접",
        style="포멀",
        season="가을",
        formality="비즈니스 캐주얼",
        constraints=["밝은 색 피하기"],
    )


@pytest.fixture
def sample_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            category="TOP",
            candidates=[
                ClothingCandidate(
                    clothes_id=101,
                    image_url="https://img.com/101.jpg",
                    category="TOP",
                    color=["흰색"],
                    style_tags=["포멀", "베이직"],
                    caption="흰색 셔츠",
                    similarity_score=0.95,
                ),
                ClothingCandidate(
                    clothes_id=102,
                    image_url="https://img.com/102.jpg",
                    category="TOP",
                    color=["네이비"],
                    style_tags=["캐주얼"],
                    caption="네이비 폴로셔츠",
                    similarity_score=0.88,
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
            ],
        ),
    ]


class TestBuildPrompt:
    def test_includes_occasion_and_style(
        self,
        composer: OutfitComposer,
        sample_parsed_query: ParsedQuery,
        sample_search_results: list[SearchResult],
    ) -> None:
        prompt = composer._build_prompt(sample_parsed_query, sample_search_results, 3)

        assert "상황: 면접" in prompt
        assert "스타일: 포멀" in prompt

    def test_includes_optional_fields(
        self,
        composer: OutfitComposer,
        sample_parsed_query: ParsedQuery,
        sample_search_results: list[SearchResult],
    ) -> None:
        prompt = composer._build_prompt(sample_parsed_query, sample_search_results, 3)

        assert "계절: 가을" in prompt
        assert "격식: 비즈니스 캐주얼" in prompt
        assert "밝은 색 피하기" in prompt

    def test_includes_candidate_info(
        self,
        composer: OutfitComposer,
        sample_parsed_query: ParsedQuery,
        sample_search_results: list[SearchResult],
    ) -> None:
        prompt = composer._build_prompt(sample_parsed_query, sample_search_results, 3)

        assert "ID: 101" in prompt
        assert "색상: 흰색" in prompt
        assert "[TOP]" in prompt
        assert "[BOTTOM]" in prompt

    def test_requests_correct_number_of_outfits(
        self,
        composer: OutfitComposer,
        sample_parsed_query: ParsedQuery,
        sample_search_results: list[SearchResult],
    ) -> None:
        prompt = composer._build_prompt(sample_parsed_query, sample_search_results, 5)
        assert "5개의 코디를 추천" in prompt


class TestBuildResponse:
    def test_parses_valid_response(
        self,
        composer: OutfitComposer,
        sample_search_results: list[SearchResult],
    ) -> None:
        candidates_map = OutfitComposer._build_candidates_map(sample_search_results)
        llm_response = OutfitCompositionLLMResponse(
            query_summary="면접용 포멀 코디",
            outfits=[
                OutfitCombination(
                    description="깔끔한 비즈니스 룩",
                    items=[
                        OutfitItemLLM(clothes_id=101, role="상의"),
                        OutfitItemLLM(clothes_id=201, role="하의"),
                    ],
                )
            ],
        )

        result = composer._build_response(llm_response, candidates_map)

        assert result.query_summary == "면접용 포멀 코디"
        assert len(result.outfits) == 1
        assert result.outfits[0].clothes_ids == [101, 201]

    def test_skips_invalid_clothes_id(
        self,
        composer: OutfitComposer,
        sample_search_results: list[SearchResult],
    ) -> None:
        candidates_map = OutfitComposer._build_candidates_map(sample_search_results)
        llm_response = OutfitCompositionLLMResponse(
            query_summary="테스트",
            outfits=[
                OutfitCombination(
                    description="테스트 코디",
                    items=[OutfitItemLLM(clothes_id=999, role="상의")],
                )
            ],
        )

        result = composer._build_response(llm_response, candidates_map)

        assert len(result.outfits) == 0


class TestEmptyResponse:
    @pytest.mark.asyncio
    async def test_returns_empty_when_no_candidates(
        self,
        composer: OutfitComposer,
        sample_parsed_query: ParsedQuery,
    ) -> None:
        empty_results: list[SearchResult] = []

        result = await composer.compose(sample_parsed_query, empty_results)

        assert len(result.outfits) == 0
        assert "찾지 못했습니다" in result.query_summary
