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
    """멀티턴 대화에서 파싱된 사용자 의도"""

    intent_type: str  # "new" | "item_change" | "style_modify" | "re_request"
    target_outfit_index: int | None  # 수정 대상 코디 인덱스 (0-based)
    target_category: str | None  # 변경할 카테고리 (예: "TOP", "BOTTOM")
    style_direction: str | None  # 스타일 변경 방향 (예: "더 캐주얼하게", "색상 밝게")


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
    intent: (
        str | None
    )  # "new_outfit" | "modify_previous" | "confirm_item" (간단한 문자열, 하위 호환)
    parsed_intent: ParsedIntent | None  # 구조화된 인텐트 (V3 신규)
    reference_outfit_id: str | None  # 수정 대상 코디 ID (modify_previous일 때)
