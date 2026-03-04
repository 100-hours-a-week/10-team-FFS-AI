"""검색+보충 서브그래프: 벡터 검색 + 평가 + 재검색 + 쇼핑 보충.

그래프 구조:
    [vector_search] → [evaluate_search]
                           │
         ├─ 전 카테고리 0건 (fast path) → [supplement_from_shop] → 반환
         │
         ├─ 충분 → 반환 (merged_candidates = search_results)
         │
         ├─ 부족 & retry < max → [relax_and_research] → [vector_search] (루프)
         │
         └─ 부족 & retry >= max → [supplement_from_shop] → 반환
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.outfit.graph.edges import should_research_or_supplement
from app.outfit.graph.nodes.search import (
    evaluate_search,
    relax_and_research,
    vector_search,
)
from app.outfit.graph.nodes.shop import supplement_from_shop
from app.outfit.graph.state import OutfitGraphState


def build_search_subgraph() -> CompiledStateGraph:
    """검색+보충 서브그래프를 빌드한다."""
    graph = StateGraph(OutfitGraphState)

    graph.add_node("vector_search", vector_search)
    graph.add_node("evaluate_search", evaluate_search)
    graph.add_node("relax_and_research", relax_and_research)
    graph.add_node("supplement_from_shop", supplement_from_shop)

    graph.add_edge(START, "vector_search")
    graph.add_edge("vector_search", "evaluate_search")

    graph.add_conditional_edges(
        "evaluate_search",
        should_research_or_supplement,
        {
            "sufficient": END,
            "retry_search": "relax_and_research",
            "supplement_from_shop": "supplement_from_shop",
        },
    )

    graph.add_edge("relax_and_research", "vector_search")
    graph.add_edge("supplement_from_shop", END)

    return graph.compile()
