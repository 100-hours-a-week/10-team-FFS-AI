from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.nodes.compose import (
    MAX_COMPOSE_RETRIES,
    adjust_compose_params,
    log_diversity,
    outfit_compose,
    validate_outfits,
)
from app.outfit.graph.state import OutfitGraphState


def should_retry_or_return(state: OutfitGraphState) -> str:
    if state.get("quality_passed"):
        return "sufficient"

    retry_count = state.get("compose_retry_count", 0)
    if retry_count < MAX_COMPOSE_RETRIES:
        return "retry"

    return "fallback"


async def set_fallback_flag(state: OutfitGraphState) -> dict:
    return {
        "fallback_used": True,
        "fallback_reason": f"코디 {len(state.get('outfits', []))}개만 생성됨",
    }


def build_compose_subgraph() -> CompiledStateGraph:
    graph = StateGraph(OutfitGraphState)

    graph.add_node("outfit_compose", outfit_compose)
    graph.add_node("validate_outfits", validate_outfits)
    graph.add_node("log_diversity", log_diversity)
    graph.add_node("adjust_compose_params", adjust_compose_params)
    graph.add_node("set_fallback_flag", set_fallback_flag)

    graph.add_edge(START, "outfit_compose")
    graph.add_edge("outfit_compose", "validate_outfits")

    graph.add_conditional_edges(
        "validate_outfits",
        should_retry_or_return,
        {
            "sufficient": "log_diversity",
            "retry": "adjust_compose_params",
            "fallback": "set_fallback_flag",
        },
    )

    graph.add_edge("log_diversity", END)
    graph.add_edge("adjust_compose_params", "outfit_compose")
    graph.add_edge("set_fallback_flag", END)

    return graph.compile()
