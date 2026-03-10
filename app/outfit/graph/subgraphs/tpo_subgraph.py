from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.nodes.tpo import (
    MAX_TPO_RETRIES,
    tpo_extract,
    tpo_fallback,
    tpo_retry,
    tpo_validate,
)
from app.outfit.graph.state import OutfitGraphState


def should_retry_or_fallback(state: OutfitGraphState) -> str:
    if state.get("quality_passed"):
        return "valid"

    retry_count = state.get("tpo_retry_count", 0)
    if retry_count < MAX_TPO_RETRIES:
        return "retry"

    return "fallback"


def build_tpo_subgraph() -> CompiledStateGraph:
    graph = StateGraph(OutfitGraphState)

    graph.add_node("tpo_extract", tpo_extract)
    graph.add_node("tpo_validate", tpo_validate)
    graph.add_node("tpo_retry", tpo_retry)
    graph.add_node("tpo_fallback", tpo_fallback)

    graph.add_edge(START, "tpo_extract")
    graph.add_edge("tpo_extract", "tpo_validate")

    graph.add_conditional_edges(
        "tpo_validate",
        should_retry_or_fallback,
        {
            "valid": END,
            "retry": "tpo_retry",
            "fallback": "tpo_fallback",
        },
    )

    graph.add_edge("tpo_retry", "tpo_extract")
    graph.add_edge("tpo_fallback", END)

    return graph.compile()
