"""TPO 서브그래프: TPO 추출 + 검증 + 재시도 + 폴백.

그래프 구조:
    [tpo_extract] → [tpo_validate]
                         │
         ├─ 유효 → 반환
         │
         ├─ 불충분 & retry < 2 → [tpo_retry] → [tpo_extract] (루프)
         │
         └─ 불충분 & retry >= 2 → [tpo_fallback] → 반환
"""

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
    """TPO 검증 결과에 따라 다음 경로를 결정한다."""
    if state.get("quality_passed"):
        return "valid"

    retry_count = state.get("tpo_retry_count", 0)
    if retry_count < MAX_TPO_RETRIES:
        return "retry"

    return "fallback"


def build_tpo_subgraph() -> CompiledStateGraph:
    """TPO 서브그래프를 빌드한다."""
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
