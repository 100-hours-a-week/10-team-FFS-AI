from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.edges import should_research_or_supplement
from app.outfit.graph.nodes.compose import outfit_compose
from app.outfit.graph.nodes.response import format_response
from app.outfit.graph.nodes.search import (
    evaluate_search,
    relax_and_research,
    vector_search,
)
from app.outfit.graph.nodes.shop import supplement_from_shop
from app.outfit.graph.nodes.tpo import tpo_extract
from app.outfit.graph.nodes.vton import vton_process
from app.outfit.graph.state import OutfitGraphState


def build_outfit_graph() -> CompiledStateGraph:
    graph = StateGraph(OutfitGraphState)

    graph.add_node("tpo_extract", tpo_extract)
    graph.add_node("vector_search", vector_search)
    graph.add_node("evaluate_search", evaluate_search)
    graph.add_node("relax_and_research", relax_and_research)
    graph.add_node("supplement_from_shop", supplement_from_shop)
    graph.add_node("outfit_compose", outfit_compose)
    graph.add_node("vton_process", vton_process)
    graph.add_node("format_response", format_response)

    graph.add_edge(START, "tpo_extract")
    graph.add_edge("tpo_extract", "vector_search")
    graph.add_edge("vector_search", "evaluate_search")

    graph.add_conditional_edges(
        "evaluate_search",
        should_research_or_supplement,
        {
            "sufficient": "outfit_compose",
            "retry_search": "relax_and_research",
            "supplement_from_shop": "supplement_from_shop",
        },
    )

    graph.add_edge("relax_and_research", "vector_search")

    graph.add_edge("supplement_from_shop", "outfit_compose")

    graph.add_edge("outfit_compose", "vton_process")
    graph.add_edge("vton_process", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
