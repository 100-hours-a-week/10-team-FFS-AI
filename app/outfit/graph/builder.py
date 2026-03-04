"""메인 그래프 빌더: 서브그래프들을 연결하여 전체 파이프라인을 구성한다.

Phase 4 구조:
    [TPO 서브그래프] → [검색+보충 서브그래프] → [조합 서브그래프]
                                                      ↓
                                            [evaluate_quality]
                                                      │
                    ┌─────────────────────────────────┼─────────────────────────────────┐
                    │                                 │                                 │
                    ▼                                 ▼                                 ▼
            (retry_compose)                       (pass)                          (fallback)
          [조합 서브그래프]               [vton_process]             [build_fallback_response]
                                                      │                                 │
                                                      ▼                                 │
                                            [format_response]  ◄────────────────────────┘
                                                      │
                                                      ▼
                                                    [END]
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.edges import should_retry_or_fallback
from app.outfit.graph.nodes.fallback import build_fallback_response
from app.outfit.graph.nodes.quality import evaluate_quality
from app.outfit.graph.nodes.response import format_response
from app.outfit.graph.nodes.vton import vton_process
from app.outfit.graph.state import OutfitGraphState
from app.outfit.graph.subgraphs import (
    build_compose_subgraph,
    build_search_subgraph,
    build_tpo_subgraph,
)


def build_outfit_graph() -> CompiledStateGraph:
    """서브그래프 기반 메인 그래프를 빌드한다."""
    graph = StateGraph(OutfitGraphState)

    tpo_subgraph = build_tpo_subgraph()
    search_subgraph = build_search_subgraph()
    compose_subgraph = build_compose_subgraph()

    # 서브그래프 노드
    graph.add_node("tpo_subgraph", tpo_subgraph)
    graph.add_node("search_subgraph", search_subgraph)
    graph.add_node("compose_subgraph", compose_subgraph)

    # Phase 4: 품질 평가 + Fallback 노드
    graph.add_node("evaluate_quality", evaluate_quality)
    graph.add_node("build_fallback_response", build_fallback_response)

    # 기존 노드
    graph.add_node("vton_process", vton_process)
    graph.add_node("format_response", format_response)

    # 엣지: TPO → 검색 → 조합 → 품질평가
    graph.add_edge(START, "tpo_subgraph")
    graph.add_edge("tpo_subgraph", "search_subgraph")
    graph.add_edge("search_subgraph", "compose_subgraph")
    graph.add_edge("compose_subgraph", "evaluate_quality")

    # Phase 4: 품질 평가 후 조건부 분기
    graph.add_conditional_edges(
        "evaluate_quality",
        should_retry_or_fallback,
        {
            "pass": "vton_process",
            "retry_compose": "compose_subgraph",
            "fallback": "build_fallback_response",
        },
    )

    # Fallback → VTON
    graph.add_edge("build_fallback_response", "vton_process")

    # VTON → 응답 포맷 → END
    graph.add_edge("vton_process", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()
