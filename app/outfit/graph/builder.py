"""build_outfit_graph: 코디 추천 LangGraph 그래프를 구성한다.

Phase 1: 순차 실행 (분기/재시도 없음)
START → tpo_extract → vector_search → outfit_compose → vton_process → format_response → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.nodes.compose import outfit_compose
from app.outfit.graph.nodes.response import format_response
from app.outfit.graph.nodes.search import vector_search
from app.outfit.graph.nodes.tpo import tpo_extract
from app.outfit.graph.nodes.vton import vton_process
from app.outfit.graph.state import OutfitGraphState


def build_outfit_graph() -> CompiledStateGraph:
    """코디 추천 파이프라인 그래프를 빌드하고 컴파일한다."""
    graph = StateGraph(OutfitGraphState)

    # 노드 추가
    graph.add_node("tpo_extract", tpo_extract)
    graph.add_node("vector_search", vector_search)
    graph.add_node("outfit_compose", outfit_compose)
    graph.add_node("vton_process", vton_process)
    graph.add_node("format_response", format_response)

    # 순차 엣지 연결
    graph.add_edge(START, "tpo_extract")
    graph.add_edge("tpo_extract", "vector_search")
    graph.add_edge("vector_search", "outfit_compose")
    graph.add_edge("outfit_compose", "vton_process")
    graph.add_edge("vton_process", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
