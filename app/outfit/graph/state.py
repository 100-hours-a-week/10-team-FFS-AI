"""OutfitGraphState: LangGraph 그래프 상태 정의.

Phase 1에서 사용하는 필드와 향후 Phase 확장을 위한 예약 필드를 포함한다.
"""

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
    """LangGraph 그래프 전체에서 공유되는 상태."""

    # --- 입력 (recommend 호출 시 설정) ---
    query: str
    user_id: int
    session_id: str | None
    trace_id: str
    weather: Weather | None
    upload_slots: list[UploadSlot]

    # --- tpo_extract 노드 출력 ---
    parsed_query: ParsedQuery
    search_queries: list[SearchQuery]
    required_categories: list[str]

    # --- vector_search 노드 출력 ---
    search_results: list[SearchResult]
    category_coverage: dict[str, int]

    # --- outfit_compose 노드 출력 ---
    outfits: list[Outfit]
    response: OutfitResponse

    # --- vton_process 노드 출력 ---
    vton_completed: bool

    # === Phase 2+: 확장 필드 ===

    # --- Shop 보충 검색 ---
    shop_results: list[SearchResult]
    merged_candidates: list[SearchResult]
    shop_supplemented: bool

    # --- 검색 재시도 ---
    search_retry_count: int
    filter_level: str  # strict/relaxed/minimal

    # --- 코디 품질 검증 ---
    compose_retry_count: int
    outfit_confidence: float
    quality_passed: bool
    quality_issues: list[str]

    # --- 에러 및 Fallback ---
    error: str | None
    fallback_used: bool
    fallback_reason: str | None
