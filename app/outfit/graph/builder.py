from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.edges import should_retry_or_fallback
from app.outfit.graph.nodes.fallback import build_fallback_response
from app.outfit.graph.nodes.quality import evaluate_quality
from app.outfit.graph.nodes.response import format_response
from app.outfit.graph.nodes.session import save_session_context
from app.outfit.graph.nodes.vton import vton_process
from app.outfit.graph.state import OutfitGraphState
from app.outfit.graph.subgraphs import (
    build_compose_subgraph,
    build_search_subgraph,
    build_tpo_subgraph,
)


def build_outfit_graph() -> CompiledStateGraph:
    graph = StateGraph(OutfitGraphState)

    tpo_subgraph = build_tpo_subgraph()
    search_subgraph = build_search_subgraph()
    compose_subgraph = build_compose_subgraph()

    graph.add_node("tpo_subgraph", tpo_subgraph)
    graph.add_node("search_subgraph", search_subgraph)
    graph.add_node("compose_subgraph", compose_subgraph)

    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("build_fallback_response", build_fallback_response)

    graph.add_node("vton_process", vton_process)
    graph.add_node("format_response", format_response)
    graph.add_node("save_session_context", save_session_context)

    graph.add_edge(START, "tpo_subgraph")
    graph.add_edge("tpo_subgraph", "search_subgraph")
    graph.add_edge("search_subgraph", "compose_subgraph")
    graph.add_edge("compose_subgraph", "evaluate_quality")

    graph.add_conditional_edges(
        "evaluate_quality",
        should_retry_or_fallback,
        {
            "pass": "vton_process",
            "retry_compose": "compose_subgraph",
            "fallback": "build_fallback_response",
        },
    )

    graph.add_edge("build_fallback_response", "vton_process")

    graph.add_edge("vton_process", "format_response")
    graph.add_edge("format_response", "save_session_context")
    graph.add_edge("save_session_context", END)

    return graph.compile()
