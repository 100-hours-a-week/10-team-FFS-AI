import json
import logging
from typing import Any

from app.common.metrics import measure_time
from app.outfit.llm_client import LLMClient
from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.schemas import ShopParsedQuery

logger = logging.getLogger(__name__)


SHOP_SYSTEM_PROMPT = """당신은 사용자의 쇼핑 검색 요청을 분석하는 AI입니다.

사용자 입력에서 다음 정보를 추출하여 JSON으로 응답하세요:

1. occasion: 상황/장소 (면접, 데이트, 출근, 여행 등). 불분명하면 "일상"
2. style: 원하는 스타일 (Y2K, 캐주얼, 미니멀, 스트릿 등). 불분명하면 "깔끔한"
3. season: 계절 (봄, 여름, 가을, 겨울). 언급 없으면 null
4. price_max: 최대 가격 (원, 정수). 언급 없으면 null. 예: "3만원 이하" → 30000
5. price_min: 최소 가격 (원, 정수). 언급 없으면 null
6. brand: 브랜드명. 언급 없으면 null
7. target_category: 찾는 아이템 카테고리 (반드시 다음 중 하나: TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC). 전체 코디 요청이면 null
8. constraints: 추가 제약사항 배열 (크롭탑, 오버핏 등 구체적 키워드)

반드시 JSON만 응답하세요. 설명이나 마크다운 없이 순수 JSON만 출력하세요.

예시 입력: "3만원 이하 Y2K 감성 크롭탑 코디"
예시 출력:
{
  "occasion": "일상",
  "style": "Y2K",
  "season": null,
  "price_max": 30000,
  "price_min": null,
  "brand": null,
  "target_category": null,
  "constraints": ["크롭탑", "Y2K 감성"]
}"""


class ShopQueryParser:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    @measure_time("shop_query_parser")
    async def parse(
        self,
        query: str,
        trace_id: str | None = None,
        user_id: int | None = None,
    ) -> ShopParsedQuery:
        messages = [
            {"role": "system", "content": SHOP_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        log_context = f"trace_id={trace_id}" if trace_id else ""
        if user_id is not None:
            log_context += (
                f" user_id={user_id}" if log_context else f"user_id={user_id}"
            )

        logger.info(f'Parsing shop query | {log_context} query="{query}"')

        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            return self._parse_response(response)

        except ShopLLMError:
            raise

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse shop LLM response | {log_context} error={e}")
            raise ShopParseError(f"Invalid LLM response format: {e}") from e

        except Exception as e:
            logger.exception(
                f"Unexpected error parsing shop query | {log_context} error={e}"
            )
            raise ShopParseError(f"Unexpected parsing error: {e}") from e

    def _parse_response(self, response: dict[str, Any]) -> ShopParsedQuery:
        content = response["choices"][0]["message"]["content"]
        data = self._extract_json(content)

        return ShopParsedQuery(
            occasion=data.get("occasion", "일상"),
            style=data.get("style", "깔끔한"),
            season=data.get("season"),
            price_max=data.get("price_max"),
            price_min=data.get("price_min"),
            brand=data.get("brand"),
            target_category=data.get("target_category"),
            constraints=data.get("constraints", []),
        )

    def _extract_json(self, content: str) -> dict[str, Any]:
        content = content.strip()

        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines)

        return json.loads(content)
