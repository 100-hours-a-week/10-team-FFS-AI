import logging

from langgraph.types import RunnableConfig

from app.outfit.graph.state import OutfitGraphState

logger = logging.getLogger(__name__)


async def outfit_compose(state: OutfitGraphState, config: RunnableConfig) -> dict:
    configurable = config.get("configurable", {})
    outfit_composer = configurable["outfit_composer"]

    parsed_query = state["parsed_query"]
    trace_id = state["trace_id"]
    user_id = state["user_id"]

    candidates = state.get("merged_candidates") or state.get("search_results", [])
    shop_supplemented = state.get("shop_supplemented", False)

    if shop_supplemented:
        logger.info(
            f"Composing with shop-supplemented candidates | "
            f"trace_id={trace_id} user_id={user_id}"
        )

    response = await outfit_composer.compose(
        parsed_query=parsed_query,
        search_results=candidates,
        trace_id=trace_id,
        user_id=user_id,
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
