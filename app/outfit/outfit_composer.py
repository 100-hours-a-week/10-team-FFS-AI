import json
import logging
import uuid

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.llm_client import LLMClient, OpenAIClient
from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
    OutfitItem,
    OutfitResponse,
    ParsedQuery,
    SearchResult,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """당신은 패션 스타일리스트입니다.
사용자의 요청과 후보 아이템들을 바탕으로 코디를 추천합니다.

규칙:
1. 반드시 주어진 후보 아이템의 clothes_id만 사용하세요.
2. 각 코디는 서로 다른 스타일/분위기를 가져야 합니다.
3. 색상 조화, TPO, 계절감을 고려하세요.
4. 3가지 코디를 추천하고, 각각에 대해 간단한 설명을 제공하세요.

JSON 형식으로만 응답하세요:
{
  "query_summary": "사용자 요청 한 줄 요약",
  "outfits": [
    {
      "description": "코디 설명",
      "items": [
        {"clothes_id": 123, "role": "상의"},
        {"clothes_id": 456, "role": "하의"}
      ]
    }
  ]
}"""


class OutfitComposer:
    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or OpenAIClient()

    async def compose(
        self,
        parsed_query: ParsedQuery,
        search_results: list[SearchResult],
        num_outfits: int = 3,
    ) -> OutfitResponse:
        if not search_results or all(len(r.candidates) == 0 for r in search_results):
            return self._empty_response(parsed_query)

        candidates_map = self._build_candidates_map(search_results)
        prompt = self._build_prompt(parsed_query, search_results, num_outfits)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            return self._parse_response(response, candidates_map)

        except LLMError:
            raise

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            raise ParseError(f"Invalid outfit response format: {e}") from e

    def _build_prompt(
        self,
        parsed: ParsedQuery,
        results: list[SearchResult],
        num_outfits: int,
    ) -> str:
        lines = [
            f"상황: {parsed.occasion}",
            f"스타일: {parsed.style}",
        ]

        if parsed.season:
            lines.append(f"계절: {parsed.season}")
        if parsed.formality:
            lines.append(f"격식: {parsed.formality}")
        if parsed.constraints:
            lines.append(f"제약사항: {', '.join(parsed.constraints)}")

        lines.append("")
        lines.append("후보 아이템:")

        for result in results:
            lines.append(f"\n[{result.category}]")
            for c in result.candidates:
                tags = ", ".join(c.style_tags) if c.style_tags else "없음"
                lines.append(
                    f"  - ID: {c.clothes_id}, 색상: {c.color or '없음'}, 스타일: {tags}"
                )

        lines.append(f"\n{num_outfits}개의 코디를 추천해주세요.")

        return "\n".join(lines)

    @staticmethod
    def _build_candidates_map(
        results: list[SearchResult],
    ) -> dict[int, ClothingCandidate]:
        candidates_map: dict[int, ClothingCandidate] = {}
        for result in results:
            for candidate in result.candidates:
                candidates_map[candidate.clothes_id] = candidate
        return candidates_map

    def _parse_response(
        self,
        response: dict,
        candidates_map: dict[int, ClothingCandidate],
    ) -> OutfitResponse:
        content = response["choices"][0]["message"]["content"]
        data = self._extract_json(content)

        outfits = []
        for outfit_data in data.get("outfits", []):
            items = []
            for item_data in outfit_data.get("items", []):
                clothes_id = item_data["clothes_id"]
                candidate = candidates_map.get(clothes_id)

                if candidate:
                    items.append(
                        OutfitItem(
                            clothes_id=clothes_id,
                            image_url=candidate.image_url,
                            category=candidate.category,
                            role=item_data.get("role", candidate.category),
                        )
                    )

            if items:
                outfits.append(
                    Outfit(
                        outfit_id=str(uuid.uuid4()),
                        description=outfit_data.get("description", ""),
                        items=items,
                    )
                )

        return OutfitResponse(
            query_summary=data.get("query_summary", "코디 추천"),
            outfits=outfits,
        )

    @staticmethod
    def _extract_json(content: str) -> dict:
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.startswith("```")]
            content = "\n".join(lines)
        return json.loads(content)

    @staticmethod
    def _empty_response(parsed: ParsedQuery) -> OutfitResponse:
        return OutfitResponse(
            query_summary=f"{parsed.occasion}에 적합한 코디를 찾지 못했습니다.",
            outfits=[],
        )
