from typing import TypedDict

from app.outfit.schemas import (
    Outfit,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
    UploadSlot,
    Weather,
)


class OutfitGraphState(TypedDict, total=False):
    query: str
    user_id: int
    session_id: str | None
    trace_id: str
    weather: Weather | None
    upload_slots: list[UploadSlot]

    parsed_query: ParsedQuery
    search_queries: list[SearchQuery]
    required_categories: list[str]

    search_results: list[SearchResult]
    category_coverage: dict[str, int]

    outfits: list[Outfit]
    response: OutfitResponse

    vton_completed: bool

    shop_results: list[SearchResult]
    merged_candidates: list[SearchResult]
    shop_supplemented: bool

    search_retry_count: int
    filter_level: str

    compose_retry_count: int
    outfit_confidence: float
    quality_passed: bool
    quality_issues: list[str]

    error: str | None
    fallback_used: bool
    fallback_reason: str | None
