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
    tpo_retry_count: int
    tpo_fallback_used: bool

    search_results: list[SearchResult]
    shop_results: list[SearchResult]
    category_coverage: dict[str, int]
    search_retry_count: int
    filter_level: str
    shop_supplemented: bool

    merged_candidates: list[SearchResult]
    outfits: list[Outfit]
    outfit_confidence: float
    compose_retry_count: int

    quality_passed: bool
    quality_issues: list[str]

    vton_completed: bool

    response: OutfitResponse

    error: str | None
    fallback_used: bool
    fallback_reason: str | None
