import json
import logging
import uuid

from app.outfit.exceptions import LLMError, ParseError
from app.outfit.llm_client import LLMClient, OpenAIClient
from app.outfit.schemas import (
    ClothingCandidate,
    Outfit,
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
        trace_id: str | None = None,
        user_id: int | None = None,
    ) -> OutfitResponse:
        log_context = f"trace_id={trace_id}" if trace_id else ""
        if user_id is not None:
            log_context += (
                f" user_id={user_id}" if log_context else f"user_id={user_id}"
            )

        if not search_results or all(len(r.candidates) == 0 for r in search_results):
            logger.info(
                f"No candidates found, returning empty response | {log_context}"
            )
            return self._empty_response(parsed_query)

        candidates_map = self._build_candidates_map(search_results)
        prompt = self._build_prompt(parsed_query, search_results, num_outfits)

        logger.info(
            f"Composing outfits | {log_context} "
            f"candidate_count={len(candidates_map)} "
            f"requested_outfits={num_outfits}"
        )

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
            outfit_response = self._parse_response(response, candidates_map)

            actual_count = len(outfit_response.outfits)
            if actual_count < num_outfits:
                logger.warning(
                    f"Insufficient outfits generated | {log_context}"
                    f"requested={num_outfits} actual={actual_count}"
                )

            for idx, outfit in enumerate(outfit_response.outfits, 1):
                items_str = ",".join(str(cid) for cid in outfit.clothes_ids)
                logger.info(
                    f"Outfit composed | {log_context} "
                    f"outfit_index={idx} "
                    f"outfit_id={outfit.outfit_id} "
                    f"items=[{items_str}] "
                    f'description="{outfit.description}"'
                )

            return outfit_response

        except LLMError:
            raise

        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse LLM response | {log_context} error={e}")
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
                    f"  - ID: {c.clothes_id}, 색상: {', '.join(c.color) if c.color else '없음'}, 스타일: {tags}"
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
            clothes_ids = []
            for item_data in outfit_data.get("items", []):
                clothes_id = item_data["clothes_id"]
                # 유효한 후보 아이템인지 확인
                if clothes_id in candidates_map:
                    clothes_ids.append(clothes_id)
                else:
                    logger.warning(f"LLM returned invalid clothes_id: {clothes_id}")

            if clothes_ids:
                outfits.append(
                    Outfit(
                        outfit_id=str(uuid.uuid4()),
                        description=outfit_data.get("description", ""),
                        clothes_ids=clothes_ids,
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
