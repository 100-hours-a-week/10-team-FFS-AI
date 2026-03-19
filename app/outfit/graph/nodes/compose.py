import asyncio
import logging
import os

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState
from app.outfit.schemas import Outfit, OutfitResponse

logger = logging.getLogger(__name__)

MIN_OUTFIT_COUNT = 3
MAX_COMPOSE_RETRIES = 2


async def outfit_compose(state: OutfitGraphState, config: RunnableConfig) -> dict:
    parsed_intent = state.get("parsed_intent")
    if parsed_intent and parsed_intent["intent_type"] == "re_request":
        trace_id = state.get("trace_id", "unknown")
        logger.info(f"Skip compose for re_request | trace_id={trace_id}")

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

    style_direction_instruction = None
    if parsed_intent and parsed_intent["intent_type"] == "style_modify":
        style_direction = parsed_intent.get("style_direction")
        if style_direction:
            trace_id = state.get("trace_id", "unknown")
            logger.info(
                f"Style modify: applying style_direction='{style_direction}' | "
                f"trace_id={trace_id}"
            )
            style_direction_instruction = (
                f"사용자가 스타일 변경을 요청했습니다: '{style_direction}'\n"
                f"이 방향에 맞게 코디를 재구성해주세요."
            )

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

    additional_instructions = None
    if style_direction_instruction and quality_feedback:
        additional_instructions = f"{quality_feedback}\n\n{style_direction_instruction}"
    elif style_direction_instruction:
        additional_instructions = style_direction_instruction
    elif quality_feedback:
        additional_instructions = quality_feedback

    response = await outfit_composer.compose(
        parsed_query=parsed_query,
        search_results=candidates,
        trace_id=trace_id,
        user_id=user_id,
        additional_instructions=additional_instructions,
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

    # Chaos injection for testing
    chaos_delay = os.getenv("CHAOS_DELAY_AFTER_COMPOSE")
    if chaos_delay:
        delay_seconds = float(chaos_delay)
        logger.warning(
            f"[CHAOS] Injecting delay after compose | "
            f"delay={delay_seconds}s | trace_id={trace_id}"
        )
        await asyncio.sleep(delay_seconds)

    return {
        "response": response,
        "outfits": response.outfits,
    }


def _calculate_jaccard(outfit_a: Outfit, outfit_b: Outfit) -> float:
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
