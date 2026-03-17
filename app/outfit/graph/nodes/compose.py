import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import Outfit, OutfitResponse

logger = logging.getLogger(__name__)

MIN_OUTFIT_COUNT = 3
MAX_COMPOSE_RETRIES = 2


async def outfit_compose(state: OutfitGraphState, config: RunnableConfig) -> dict:
    # 인텐트 체크: re_request는 조합 스킵, previous_outfits 반환
    parsed_intent = state.get("parsed_intent")
    if parsed_intent and parsed_intent["intent_type"] == "re_request":
        trace_id = state.get("trace_id", "unknown")
        logger.info(f"Skip compose for re_request | trace_id={trace_id}")

        # previous_outfits를 outfits로 복사
        session_data = state.get("session_data")
        previous_outfits = session_data.previous_outfits if session_data else []

        if previous_outfits:
            logger.info(
                f"Returning previous outfits | trace_id={trace_id} "
                f"count={len(previous_outfits)}"
            )
            return {"outfits": previous_outfits}
        else:
            logger.warning(
                f"No previous outfits found for re_request | trace_id={trace_id}"
            )
            return {"outfits": []}

    # style_modify: 스타일 방향 프롬프트에 반영
    if parsed_intent and parsed_intent["intent_type"] == "style_modify":
        style_direction = parsed_intent.get("style_direction")
        if style_direction:
            trace_id = state.get("trace_id", "unknown")
            logger.info(
                f"Style modify: applying style_direction='{style_direction}' | "
                f"trace_id={trace_id}"
            )
            # TODO: Phase 3 - style_direction을 조합 프롬프트에 반영
            # 현재는 기본 조합 로직 사용

    configurable = config.get("configurable", {})
    outfit_composer = configurable["outfit_composer"]

    trace_id = state["trace_id"]
    user_id = state["user_id"]

    parsed_query = state.get("parsed_query")
    if not parsed_query:
        error_msg = state.get("error", "알 수 없는 오류")
        logger.warning(
            f"Skipping compose due to missing parsed_query | "
            f"trace_id={trace_id} error={error_msg}"
        )
        return {
            "response": OutfitResponse(
                query_summary=f"요청 처리 중 오류가 발생했습니다: {error_msg}",
                outfits=[],
            ),
            "outfits": [],
        }

    candidates = state.get("merged_candidates") or state.get("search_results", [])
    shop_supplemented = state.get("shop_supplemented", False)

    if shop_supplemented:
        logger.info(
            f"Composing with shop-supplemented candidates | "
            f"trace_id={trace_id} user_id={user_id}"
        )

    quality_feedback = None
    critical_issues = state.get("critical_issues", [])
    if critical_issues:
        quality_feedback = (
            "이전 추천에서 다음 문제가 발견되었습니다: "
            + ", ".join(critical_issues)
            + "\n이 문제를 피해서 코디를 다시 구성해주세요."
        )
        logger.info(
            f"Retrying compose with critical feedback | "
            f"trace_id={trace_id} critical_issues={critical_issues}"
        )

    response = await outfit_composer.compose(
        parsed_query=parsed_query,
        search_results=candidates,
        trace_id=trace_id,
        user_id=user_id,
        additional_instructions=quality_feedback,
    )

    outfits_detail = []
    for idx, outfit in enumerate(response.outfits, 1):
        items_str = ",".join(str(cid) for cid in outfit.clothes_ids)
        desc_preview = outfit.description[:50] if outfit.description else "N/A"
        outfits_detail.append(
            f"[outfit_{idx}: id={outfit.outfit_id} "
            f"items=[{items_str}] "
            f'desc="{desc_preview}"]'
        )

    logger.info(
        f"Generated outfit recommendations | trace_id={trace_id} "
        f"user_id={user_id} "
        f"outfit_count={len(response.outfits)} "
        f"shop_supplemented={shop_supplemented} "
        f"outfits={' '.join(outfits_detail)}"
    )

    return {
        "response": response,
        "outfits": response.outfits,
    }


def _calculate_jaccard(outfit_a: Outfit, outfit_b: Outfit) -> float:
    """두 코디의 clothes_ids Jaccard 유사도를 계산한다."""
    set_a = set(outfit_a.clothes_ids)
    set_b = set(outfit_b.clothes_ids)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


async def validate_outfits(state: OutfitGraphState, config: RunnableConfig) -> dict:
    trace_id = state.get("trace_id", "unknown")
    outfits = state.get("outfits", [])
    outfit_count = len(outfits)

    if outfit_count >= MIN_OUTFIT_COUNT:
        logger.info(
            f"Outfit validation passed | trace_id={trace_id} "
            f"outfit_count={outfit_count}"
        )
        return {"quality_passed": True}

    logger.warning(
        f"Outfit validation failed: insufficient count | trace_id={trace_id} "
        f"outfit_count={outfit_count} min_required={MIN_OUTFIT_COUNT}"
    )
    return {"quality_passed": False}


async def log_diversity(state: OutfitGraphState, config: RunnableConfig) -> dict:
    trace_id = state.get("trace_id", "unknown")
    outfits = state.get("outfits", [])

    if len(outfits) < 2:
        logger.info(
            f"Diversity logging skipped: insufficient outfits | "
            f"trace_id={trace_id} outfit_count={len(outfits)}"
        )
        return {}

    jaccard_scores: list[float] = []
    for i in range(len(outfits)):
        for j in range(i + 1, len(outfits)):
            score = _calculate_jaccard(outfits[i], outfits[j])
            jaccard_scores.append(score)

    jaccard_max = max(jaccard_scores) if jaccard_scores else 0.0
    jaccard_avg = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0.0

    logger.info(
        f"Outfit diversity metrics | trace_id={trace_id} "
        f"jaccard_max={jaccard_max:.3f} jaccard_avg={jaccard_avg:.3f} "
        f"outfit_count={len(outfits)}"
    )

    return {"outfit_confidence": 1.0 - jaccard_avg}


async def adjust_compose_params(
    state: OutfitGraphState, config: RunnableConfig
) -> dict:
    trace_id = state.get("trace_id", "unknown")
    current_retry = state.get("compose_retry_count", 0)

    logger.info(
        f"Adjusting compose params for retry | trace_id={trace_id} "
        f"retry_count={current_retry + 1}"
    )

    return {
        "compose_retry_count": current_retry + 1,
        "quality_passed": False,
    }
