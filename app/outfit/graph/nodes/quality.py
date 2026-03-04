import logging
from collections import Counter

from langfuse import get_client
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


FORMAL_GROUP = {"포멀", "세미포멀", "비즈니스", "오피스", "드레시"}
CASUAL_GROUP = {"캐주얼", "스트리트", "스포티", "액티브", "레저"}


def build_candidates_map(
    merged_candidates: list[SearchResult],
) -> dict[int, ClothingCandidate]:
    candidates_map: dict[int, ClothingCandidate] = {}
    for result in merged_candidates:
        for candidate in result.candidates:
            candidates_map[candidate.clothes_id] = candidate
    return candidates_map


def check_item_validity(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[bool, list[str]]:
    issues: list[str] = []
    invalid_ids = [cid for cid in outfit.clothes_ids if cid not in candidates_map]
    if invalid_ids:
        issues.append(f"존재하지 않는 아이템 ID: {invalid_ids}")
        return False, issues
    return True, issues


def check_category_duplication(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str], bool]:
    categories: list[str] = []
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if candidate:
            categories.append(candidate.category)

    counter = Counter(categories)
    duplicates = {cat: cnt for cat, cnt in counter.items() if cnt > 1}

    if duplicates:
        dup_str = ", ".join(f"{cat}={cnt}개" for cat, cnt in duplicates.items())
        return 0.3, [f"카테고리 중복 ({dup_str})"], True  # Critical
    return 0.0, [], False


def check_color_harmony(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str], bool]:
    chromatic_colors: set[str] = set()
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if candidate and candidate.color:
            for c in candidate.color:
                if c not in ACHROMATIC_COLORS:
                    chromatic_colors.add(c)

    if len(chromatic_colors) > 4:
        return (
            0.2,
            [f"유채색 {len(chromatic_colors)}색 초과 ({', '.join(chromatic_colors)})"],
            False,
        )
    return 0.0, [], False


def check_formality_consistency(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
) -> tuple[float, list[str], bool]:
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
        return 0.3, ["포멀+캐주얼 아이템 혼합"], False  # Warning
    return 0.0, [], False


def check_season_compatibility(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
    request_season: str | None,
) -> tuple[float, list[str], bool]:
    if not request_season:
        return 0.0, [], False

    mismatches: list[str] = []
    for cid in outfit.clothes_ids:
        candidate = candidates_map.get(cid)
        if not candidate or not candidate.season:
            continue

        if "사계절" in candidate.season:
            continue

        if request_season not in candidate.season:
            mismatches.append(f"{candidate.category}({', '.join(candidate.season)})")

    if mismatches:
        return (
            0.2,
            [f"계절 불일치 (요청: {request_season}, 불일치: {', '.join(mismatches)})"],
            False,
        )
    return 0.0, [], False


def calculate_outfit_confidence(
    outfit: Outfit,
    candidates_map: dict[int, ClothingCandidate],
    request_season: str | None,
) -> tuple[float, list[str], list[str]]:
    is_valid, validity_issues = check_item_validity(outfit, candidates_map)
    if not is_valid:
        return 0.0, validity_issues, validity_issues  # item_validity는 critical

    total_penalty = 0.0
    all_issues: list[str] = []
    critical_issues: list[str] = []

    penalty, issues, is_critical = check_category_duplication(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)
    if is_critical:
        critical_issues.extend(issues)

    penalty, issues, is_critical = check_color_harmony(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)
    if is_critical:
        critical_issues.extend(issues)

    penalty, issues, is_critical = check_formality_consistency(outfit, candidates_map)
    total_penalty += penalty
    all_issues.extend(issues)
    if is_critical:
        critical_issues.extend(issues)

    penalty, issues, is_critical = check_season_compatibility(
        outfit, candidates_map, request_season
    )
    total_penalty += penalty
    all_issues.extend(issues)
    if is_critical:
        critical_issues.extend(issues)

    confidence = max(0.0, 1.0 - total_penalty)
    return confidence, all_issues, critical_issues


async def evaluate_quality(state: OutfitGraphState, config: RunnableConfig) -> dict:
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
            "critical_issues": ["코디가 생성되지 않음"],
            "outfit_confidence": 0.0,
            "quality_retry_count": quality_retry_count + 1,
        }

    candidates_map = build_candidates_map(merged_candidates)

    request_season = parsed_query.season if parsed_query else None

    valid_outfits: list[Outfit] = []
    all_issues: list[str] = []
    all_critical_issues: list[str] = []
    confidences: list[float] = []

    for outfit in outfits:
        confidence, issues, critical_issues = calculate_outfit_confidence(
            outfit, candidates_map, request_season
        )

        if confidence == 0.0 and issues:
            logger.warning(
                f"Removing outfit {outfit.outfit_id}: {issues} | trace_id={trace_id}"
            )
            all_issues.extend(issues)
            all_critical_issues.extend(critical_issues)
            continue

        valid_outfits.append(outfit)
        confidences.append(confidence)
        all_issues.extend(issues)
        all_critical_issues.extend(critical_issues)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    has_critical = len(all_critical_issues) > 0
    quality_passed = (
        len(valid_outfits) >= 1 and avg_confidence >= 0.5 and not has_critical
    )

    logger.info(
        f"Quality evaluation | "
        f"trace_id={trace_id} "
        f"passed={quality_passed} "
        f"avg_confidence={avg_confidence:.2f} "
        f"valid_outfits={len(valid_outfits)}/{len(outfits)} "
        f"critical_issues={all_critical_issues} "
        f"warning_issues={[i for i in all_issues if i not in all_critical_issues]}"
    )

    try:
        langfuse = get_client()
        if langfuse:
            langfuse.create_score(
                trace_id=trace_id,
                name=f"outfit_confidence_attempt_{quality_retry_count}",
                value=avg_confidence,
                data_type="NUMERIC",
                comment=f"valid={len(valid_outfits)}/{len(outfits)}, issues={len(all_issues)}",
            )
            if all_critical_issues:
                langfuse.create_score(
                    trace_id=trace_id,
                    name=f"critical_issues_attempt_{quality_retry_count}",
                    value=len(all_critical_issues),
                    data_type="NUMERIC",
                    comment=", ".join(all_critical_issues),
                )
            langfuse.flush()
    except Exception as e:
        logger.warning(
            f"Langfuse score recording failed | trace_id={trace_id} error={e}"
        )

    return {
        "quality_passed": quality_passed,
        "quality_issues": all_issues,
        "critical_issues": all_critical_issues,
        "outfit_confidence": round(avg_confidence, 2),
        "outfits": valid_outfits,
        "quality_retry_count": quality_retry_count + 1,
    }
