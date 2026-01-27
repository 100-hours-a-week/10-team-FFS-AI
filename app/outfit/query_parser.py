import json
import logging
from typing import Any

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.llm_client import LLMClient
from app.outfit.schemas import ParsedQuery, ReferenceItem

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 사용자의 코디 요청을 분석하는 AI입니다.

사용자 입력에서 다음 정보를 추출하여 JSON으로 응답하세요:

1. occasion: 상황/장소 (면접, 데이트, 출근, 결혼식, 여행 등). 불분명하면 "일상"
2. style: 원하는 스타일 (깔끔한, 캐주얼, 포멀, 스트릿 등). 불분명하면 "깔끔한"
3. season: 계절 (봄, 여름, 가을, 겨울). 언급 없으면 null
4. formality: 격식 수준 (포멀, 세미포멀, 캐주얼). 추론 가능하면 포함
5. reference_item: 사용자가 언급한 기준 아이템 (있는 경우만)
   - category: 카테고리 (코트, 셔츠, 바지 등)
   - color: 색상
   - style: 스타일 (오버핏 등)
   - description: 기타 설명
6. target_category: 찾고 있는 아이템 카테고리 (바지, 신발 등). 전체 코디 요청이면 null
7. constraints: 추가 제약사항 배열 (밝은 색으로, 편한 신발 등)

반드시 JSON만 응답하세요. 설명이나 마크다운 없이 순수 JSON만 출력하세요.

예시 입력: "내일 면접인데 검정 코트에 어울리는 바지 추천해줘"
예시 출력:
{
  "occasion": "면접",
  "style": "포멀",
  "season": null,
  "formality": "포멀",
  "reference_item": {
    "category": "코트",
    "color": "검정",
    "style": null,
    "description": null
  },
  "target_category": "바지",
  "constraints": []
}"""


class QueryParser:
    """사용자 쿼리를 파싱하여 구조화된 정보 추출"""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    async def parse(self, query: str) -> ParsedQuery:
        """사용자 쿼리를 파싱합니다."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        logger.info(f"Parsing query: {query}")

        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=500,
            )
            return self._parse_response(response)

        except LLMError:
            raise

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ParseError(f"Invalid LLM response format: {e}") from e

        except Exception as e:
            logger.exception(f"Unexpected error parsing query: {e}")
            raise ParseError(f"Unexpected parsing error: {e}") from e

    def _parse_response(self, response: dict[str, Any]) -> ParsedQuery:
        """LLM 응답에서 ParsedQuery 추출"""
        content = response["choices"][0]["message"]["content"]
        data = self._extract_json(content)

        # reference_item 처리
        reference_item = None
        if data.get("reference_item"):
            reference_item = ReferenceItem(**data["reference_item"])

        return ParsedQuery(
            occasion=data.get("occasion", "일상"),
            style=data.get("style", "깔끔한"),
            season=data.get("season"),
            formality=data.get("formality"),
            reference_item=reference_item,
            target_category=data.get("target_category"),
            constraints=data.get("constraints", []),
        )

    def _extract_json(self, content: str) -> dict[str, Any]:
        """문자열에서 JSON 추출 (마크다운 코드블록 처리)"""
        content = content.strip()

        # 마크다운 코드블록 제거
        if content.startswith("```"):
            lines = content.split("\n")
            # 첫 줄(```json)과 마지막 줄(```) 제거
            lines = [l for l in lines if not l.startswith("```")]
            content = "\n".join(lines)

        return json.loads(content)