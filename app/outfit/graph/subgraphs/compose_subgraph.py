"""조합 서브그래프: 코디 조합 + 검증 + 재시도.

그래프 구조:
    [outfit_compose] → [validate_outfits]
                            │
         ├─ 3세트 이상 → [log_diversity] → 반환
         │
         ├─ 3세트 미만 & retry < 2 → [adjust_compose_params] → [outfit_compose] (루프)
         │
         └─ 3세트 미만 & retry >= 2 → 있는 만큼 반환 + fallback 플래그
"""

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
    """코디 검증 결과에 따라 다음 경로를 결정한다."""
    if state.get("quality_passed"):
        return "sufficient"

    retry_count = state.get("compose_retry_count", 0)
    if retry_count < MAX_COMPOSE_RETRIES:
        return "retry"

    return "fallback"


async def set_fallback_flag(state: OutfitGraphState) -> dict:
    """최대 재시도 후에도 부족하면 fallback 플래그를 설정한다."""
    return {
        "fallback_used": True,
        "fallback_reason": f"코디 {len(state.get('outfits', []))}개만 생성됨",
    }


def build_compose_subgraph() -> CompiledStateGraph:
    """조합 서브그래프를 빌드한다."""
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
