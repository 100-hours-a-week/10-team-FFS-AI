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
    optional_categories: list[str]  # 있으면 활용, 없어도 코디 성립 (봄/가을 OUTER 등)
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

    # 멀티턴 관련 필드
    session_data: SessionData | None  # 세션 데이터
    intent: str | None  # "new_outfit" | "modify_previous" | "confirm_item"
    reference_outfit_id: str | None  # 수정 대상 코디 ID (modify_previous일 때)
