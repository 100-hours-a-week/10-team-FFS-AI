from typing import TypedDict

from app.outfit.schemas import (
    Outfit,
    OutfitResponse,
    ParsedQuery,
    SearchQuery,
    SearchResult,
    SessionData,
    UploadSlot,
    Weather,
)


class ParsedIntent(TypedDict):
    intent_type: str
    target_outfit_index: int | None  # 수정 대상 코디 인덱스 (0-based)
    target_category: str | None
    style_direction: str | None


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
    optional_categories: list[str]
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
    critical_issues: list[str]
    quality_retry_count: int

    vton_completed: bool

    response: OutfitResponse

    error: str | None
    fallback_used: bool
    fallback_reason: str | None

    session_data: SessionData | None
    intent: str | None
    parsed_intent: ParsedIntent | None
    reference_outfit_id: str | None
