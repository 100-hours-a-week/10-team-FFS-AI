"""품질 평가 규칙 함수들.

코디 조합 결과를 규칙 기반으로 평가하고 Confidence Score를 산출한다.
"""

import logging
from collections import Counter

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import ClothingCandidate, Outfit, SearchResult

logger = logging.getLogger(__name__)

# 무채색 목록 (색상 조화 규칙에서 제외)
ACHROMATIC_COLORS = {
    "블랙",
    "화이트",
    "그레이",
    "아이보리",
    "베이지",
    "크림",
    "실버",
    "골드",
    "검정",
    "흰색",
    "회색",
    "검은색",
    "흰",
    "검정색",
    "차콜",
}

# Formality 그룹 분류
FORMAL_GROUP = {"포멀", "세미포멀", "비즈니스", "오피스", "드레시"}
CASUAL_GROUP = {"캐주얼", "스트리트", "스포티", "액티브", "레저"}


def build_candidates_map(
    merged_candidates: list[SearchResult],
) -> dict[int, ClothingCandidate]:
    """merged_candidates의 모든 아이템을 clothes_id → ClothingCandidate로 매핑."""
    candidates_map: dict[int, ClothingCandidate] = {}
    for result in merged_candidates:
        for candidate in result.candidates:
            candidates_map[candidate.clothes_id] = candidate
    return candidates_map


def check_item_validity(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[bool, list[str]]:
    """아이템 유효성 검증.

    LLM이 hallucination으로 존재하지 않는 clothes_id를 반환하는 경우 차단.

    Returns:
        (is_valid, issues) - is_valid=False이면 해당 코디 전체를 제거한다.
    """
    issues: list[str] = []
    invalid_ids = [cid for cid in outfit.clothes_ids if cid not in candidates_map]
    if invalid_ids:
        issues.append(f"존재하지 않는 아이템 ID: {invalid_ids}")
        return False, issues
    return True, issues


def check_category_duplication(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str]]:
    """카테고리 중복 검사.

    같은 카테고리 아이템이 2개 이상 포함된 코디 감지.

    Returns:
        (penalty, issues) - 감점: -0.3
    """
    categories: list[str] = []
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if candidate:
            categories.append(candidate.category)

    counter = Counter(categories)
    duplicates = {cat: cnt for cat, cnt in counter.items() if cnt > 1}

    if duplicates:
        dup_str = ", ".join(f"{cat}={cnt}개" for cat, cnt in duplicates.items())
        return 0.3, [f"카테고리 중복 ({dup_str})"]
    return 0.0, []


def check_color_harmony(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str]]:
    """색상 조화 검사.

    코디 내 유채색이 4색을 초과하면 산만함.

    Returns:
        (penalty, issues) - 감점: -0.2
    """
    chromatic_colors: set[str] = set()
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if candidate and candidate.color:
            for c in candidate.color:
                if c not in ACHROMATIC_COLORS:
                    chromatic_colors.add(c)

    if len(chromatic_colors) > 4:
        return 0.2, [
            f"유채색 {len(chromatic_colors)}색 초과 ({', '.join(chromatic_colors)})"
        ]
    return 0.0, []


def check_formality_consistency(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str]]:
    """TPO/Formality 일관성 검사.

    포멀 아이템과 캐주얼 아이템이 혼합된 코디 감지.

    Returns:
        (penalty, issues) - 감점: -0.3
    """
    has_formal = False
    has_casual = False

    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if not candidate or not candidate.formality:
            continue
        formality_lower = candidate.formality.strip()
        if formality_lower in FORMAL_GROUP:
            has_formal = True
        elif formality_lower in CASUAL_GROUP:
            has_casual = True

    if has_formal and has_casual:
        return 0.3, ["포멀+캐주얼 아이템 혼합"]
    return 0.0, []


def check_season_compatibility(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
    request_season: str | None,
) -> tuple[float, list[str]]:
    """계절 적합성 검사.

    요청 계절과 아이템 계절이 불일치하는 경우 감지.

    Returns:
        (penalty, issues) - 감점: -0.2
    """
    if not request_season:
        return 0.0, []

    mismatches: list[str] = []
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if not candidate or not candidate.season:
            continue
        # "사계절"이면 항상 통과
        if "사계절" in candidate.season:
            continue
        # 교집합 확인
        if request_season not in candidate.season:
            mismatches.append(f"{candidate.category}({', '.join(candidate.season)})")

    if mismatches:
        return 0.2, [
            f"계절 불일치 (요청: {request_season}, 불일치: {', '.join(mismatches)})"
        ]
    return 0.0, []


def calculate_outfit_confidence(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
    request_season: str | None,
) -> tuple[float, list[str]]:
    """단일 코디에 대한 confidence score와 issues를 계산.

    Returns:
        (confidence, issues)
        - confidence: 0.0 ~ 1.0
        - issues: 위반 규칙 설명 문자열 목록
    """
    # 1. 아이템 유효성 (실패 시 즉시 반환)
    is_valid, validity_issues = check_item_validity(outfit, candidates_map)
    if not is_valid:
        return 0.0, validity_issues  # confidence 0 → 코디 제거 대상

    total_penalty = 0.0
    all_issues: list[str] = []

    # 2. 카테고리 중복
    penalty, issues = check_category_duplication(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)

    # 3. 색상 조화
    penalty, issues = check_color_harmony(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)

    # 4. Formality 일관성
    penalty, issues = check_formality_consistency(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)

    # 5. 계절 적합성
    penalty, issues = check_season_compatibility(outfit, candidates_map, request_season)
    total_penalty += penalty
    all_issues.extend(issues)

    confidence = max(0.0, 1.0 - total_penalty)
    return confidence, all_issues


async def evaluate_quality(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """코디 품질을 규칙 기반으로 평가한다.

    읽는 State 필드: outfits, parsed_query, merged_candidates, quality_retry_count
    쓰는 State 필드: quality_passed, quality_issues, outfit_confidence, outfits, quality_retry_count
    """
    outfits = state.get("outfits", [])
    merged_candidates = state.get("merged_candidates", [])
    parsed_query = state.get("parsed_query")
    quality_retry_count = state.get("quality_retry_count", 0)
    trace_id = state.get("trace_id", "unknown")

    if not outfits:
        logger.warning(f"No outfits to evaluate | trace_id={trace_id}")
        return {
            "quality_passed": False,
            "quality_issues": ["코디가 생성되지 않음"],
            "outfit_confidence": 0.0,
            "quality_retry_count": quality_retry_count + 1,
        }

    # candidates_map 구축
    candidates_map = build_candidates_map(merged_candidates)

    # 요청 계절 추출
    request_season = parsed_query.season if parsed_query else None

    # 각 코디 평가
    valid_outfits: list[Outfit] = []
    all_issues: list[str] = []
    confidences: list[float] = []

    for outfit in outfits:
        confidence, issues = calculate_outfit_confidence(
            outfit, candidates_map, request_season
        )

        if confidence == 0.0 and issues:
            # 아이템 유효성 실패 → 코디 제거
            logger.warning(
                f"Removing outfit {outfit.outfit_id}: {issues} | trace_id={trace_id}"
            )
            all_issues.extend(issues)
            continue

        valid_outfits.append(outfit)
        confidences.append(confidence)
        all_issues.extend(issues)

    # 전체 confidence = 유효한 코디들의 평균
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    # 통과 기준: 유효한 코디가 1개 이상이고, 평균 confidence >= 0.5
    quality_passed = len(valid_outfits) >= 1 and avg_confidence >= 0.5

    logger.info(
        f"Quality evaluation | "
        f"trace_id={trace_id} "
        f"passed={quality_passed} "
        f"avg_confidence={avg_confidence:.2f} "
        f"valid_outfits={len(valid_outfits)}/{len(outfits)} "
        f"issues={all_issues}"
    )

    return {
        "quality_passed": quality_passed,
        "quality_issues": all_issues,
        "outfit_confidence": round(avg_confidence, 2),
        "outfits": valid_outfits,
        "quality_retry_count": quality_retry_count + 1,
    }
