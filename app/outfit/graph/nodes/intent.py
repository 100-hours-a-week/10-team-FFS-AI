"""멀티턴 대화 인텐트 판별 노드"""

from __future__ import annotations

import logging

from langgraph.types import RunnableConfig

from app.common.llm_schemas import IntentDetectionLLMResponse
from app.outfit.exceptions import LLMError
from app.outfit.graph.state import OutfitGraphState, ParsedIntent
from app.outfit.schemas import SessionData

logger = logging.getLogger(__name__)


INTENT_SYSTEM_PROMPT = """당신은 사용자의 코디 추천 대화에서 의도를 파악하는 AI입니다.

사용자의 현재 메시지와 이전 대화 맥락을 분석하여 다음 4가지 의도 중 하나를 판별하세요:

1. **new**: 새로운 코디 추천 요청
   예시: "면접 때 입을 코디 추천해줘", "데이트룩 보여줘"

2. **item_change**: 특정 카테고리의 아이템 교체 요청
   예시: "두 번째 코디에서 바지만 바꿔줘", "상의를 달리해줘", "신발 다른 걸로"

3. **style_modify**: 스타일 방향만 수정 (아이템 교체 없이 재조합)
   예시: "더 캐주얼하게", "포멀하게 바꿔줘", "색상 밝게"

4. **re_request**: 이전 추천 결과 다시 보기
   예시: "첫 번째 코디 다시 보여줘", "아까 그거 다시"

## 판별 규칙

- 카테고리 언급("바지", "상의", "신발" 등) → **item_change**
- 코디 번호 + 교체 의도("첫 번째 코디에서 XX만") → **item_change** + target_outfit_index
- 스타일 변경만 언급("더 OO하게") → **style_modify**
- 재요청 표현("다시", "아까") → **re_request**
- 위 패턴 없으면 → **new**

## target_outfit_index 추출 규칙

- "첫 번째 코디" → 0
- "두 번째 코디" → 1
- "세 번째 코디" → 2
- 번호 언급 없음 → None (= 전체 코디 대상)

## target_category 매핑

- 바지, 하의, 팬츠 → BOTTOM
- 상의, 티셔츠, 셔츠 → TOP
- 겉옷, 아우터, 코트, 자켓 → OUTER
- 신발 → SHOES
- 원피스, 드레스 → DRESS
- 악세사리, 모자, 가방 → ACCESSORY
- 언급 없음 → None (= 전체 카테고리 대상)

대화 맥락이 없으면 항상 **new**로 판별하세요."""


async def detect_intent(state: OutfitGraphState, config: RunnableConfig) -> dict:
    """사용자 의도를 LLM으로 분석하여 ParsedIntent 반환"""
    query = state["query"]
    session_data = state.get("session_data")
    trace_id = state.get("trace_id", "unknown")

    default_intent: ParsedIntent = {
        "intent_type": "new",
        "target_outfit_index": None,
        "target_category": None,
        "style_direction": None,
    }

    # 세션이 없으면 무조건 new
    if not session_data or not session_data.history:
        logger.info(f"Intent: new (no session) | trace_id={trace_id}")
        return {"intent": "new_outfit", "parsed_intent": default_intent}

    # LLM으로 인텐트 판별
    llm_client = config["configurable"].get("llm_client")
    if not llm_client:
        # llm_client가 config에 없으면 폴백
        logger.warning(
            f"llm_client not in config, fallback to new | trace_id={trace_id}"
        )
        return {"intent": "new_outfit", "parsed_intent": default_intent}

    # 대화 히스토리 구성
    messages = _build_intent_messages(query, session_data)

    try:
        response: IntentDetectionLLMResponse = await llm_client.chat_completion(
            messages=messages,
            response_format=IntentDetectionLLMResponse,
            temperature=0.0,
            max_tokens=200,
        )

        parsed_intent: ParsedIntent = {
            "intent_type": response.intent_type,
            "target_outfit_index": response.target_outfit_index,
            "target_category": response.target_category,
            "style_direction": response.style_direction,
        }

        intent_str = f"{response.intent_type}_outfit"
        if response.intent_type == "item_change":
            intent_str = "modify_previous"
        elif response.intent_type == "re_request":
            intent_str = "confirm_item"

        logger.info(
            f"Intent: {response.intent_type} | "
            f"outfit_idx={response.target_outfit_index} | "
            f"category={response.target_category} | "
            f"trace_id={trace_id}"
        )

        return {"intent": intent_str, "parsed_intent": parsed_intent}

    except LLMError as e:
        logger.error(f"LLM error in intent detection | trace_id={trace_id} | error={e}")

        return {"intent": "new_outfit", "parsed_intent": default_intent}

    except Exception:
        logger.exception(f"Unexpected error in intent detection | trace_id={trace_id}")
        return {"intent": "new_outfit", "parsed_intent": default_intent}


def _build_intent_messages(query: str, session_data: SessionData) -> list[dict]:
    messages = [{"role": "system", "content": INTENT_SYSTEM_PROMPT}]

    for turn in session_data.history[-5:]:
        messages.append(
            {
                "role": turn.role,
                "content": turn.content,
            }
        )

    messages.append({"role": "user", "content": query})

    return messages
