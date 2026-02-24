import json
import logging
import uuid

from app.common.metrics import measure_time
from app.outfit.llm_client import LLMClient, OpenAIClient
from app.shop.exceptions import ShopLLMError, ShopParseError
from app.shop.schemas import (
    ProductCandidate,
    ProductSearchResult,
    ShopOutfit,
    ShopParsedQuery,
    ShopProduct,
    ShopSearchResponse,
)

logger = logging.getLogger(__name__)


SHOP_COMPOSER_PROMPT = """당신은 쇼핑 코디 스타일리스트입니다.
사용자의 요청과 검색된 상품을 바탕으로 코디 조합을 추천합니다.

규칙:
1. 반드시 주어진 후보 상품의 product_id만 사용하세요.
2. 각 코디는 서로 다른 스타일/분위기를 가져야 합니다.
3. 색상 조화, TPO, 계절감, 가격대를 고려하세요.
4. 3가지 코디를 추천하세요.

JSON 형식으로만 응답하세요:
{
  "query_summary": "사용자 요청 한 줄 요약 (예: Y2K 감성의 크롭탑 코디입니다)",
  "outfits": [
    {
      "items": [
        {"product_id": "prod_001"},
        {"product_id": "prod_002"}
      ]
    }
  ]
}"""


class ShopComposer:


    def __init__(
        self, llm_client: LLMClient | None = None
    ) -> None:
        self.llm_client = llm_client or OpenAIClient()

    @measure_time("shop_composer")
    async def compose(
        self,
        parsed_query: ShopParsedQuery,
        search_results: list[ProductSearchResult],
        num_outfits: int = 3,
        trace_id: str | None = None,
        user_id: int | None = None,
    ) -> ShopSearchResponse:

        log_context = (
            f"trace_id={trace_id}" if trace_id else ""
        )
        if user_id is not None:
            log_context += (
                f" user_id={user_id}"
                if log_context
                else f"user_id={user_id}"
            )


        if not search_results or all(
            len(r.candidates) == 0 for r in search_results
        ):
            logger.info(
                f"No product candidates, returning empty | "
                f"{log_context}"
            )
            return self._empty_response(parsed_query)

        candidates_map = self._build_candidates_map(
            search_results
        )
        prompt = self._build_prompt(
            parsed_query, search_results, num_outfits
        )

        logger.info(
            f"Composing shop outfits | {log_context} "
            f"candidate_count={len(candidates_map)} "
            f"requested_outfits={num_outfits}"
        )

        messages = [
            {"role": "system", "content": SHOP_COMPOSER_PROMPT},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self.llm_client.chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=1500,
            )
            outfit_response = self._parse_response(
                response, candidates_map
            )

            actual_count = len(outfit_response.outfits)
            if actual_count < num_outfits:
                logger.warning(
                    f"Insufficient shop outfits | "
                    f"{log_context} "
                    f"requested={num_outfits} "
                    f"actual={actual_count}"
                )

            return outfit_response

        except ShopLLMError:
            raise

        except (
            KeyError,
            IndexError,
            json.JSONDecodeError,
        ) as e:
            logger.error(
                f"Failed to parse shop LLM response | "
                f"{log_context} error={e}"
            )
            raise ShopParseError(
                f"Invalid shop outfit response: {e}"
            ) from e

    def _build_prompt(
        self,
        parsed: ShopParsedQuery,
        results: list[ProductSearchResult],
        num_outfits: int,
    ) -> str:

        lines = [
            f"상황: {parsed.occasion}",
            f"스타일: {parsed.style}",
        ]

        if parsed.season:
            lines.append(f"계절: {parsed.season}")
        if parsed.price_max:
            lines.append(f"최대 가격: {parsed.price_max:,}원")
        if parsed.price_min:
            lines.append(f"최소 가격: {parsed.price_min:,}원")
        if parsed.brand:
            lines.append(f"브랜드: {parsed.brand}")
        if parsed.constraints:
            lines.append(
                f"제약사항: {', '.join(parsed.constraints)}"
            )

        lines.append("")
        lines.append("후보 상품:")

        for result in results:
            lines.append(f"\n[{result.category}]")
            for c in result.candidates:
                tags = (
                    ", ".join(c.style_tags)
                    if c.style_tags
                    else "없음"
                )
                lines.append(
                    f"  - ID: {c.product_id}, "
                    f"이름: {c.title}, "
                    f"브랜드: {c.brand or '없음'}, "
                    f"가격: {c.price:,}원, "
                    f"스타일: {tags}"
                )

        lines.append(
            f"\n{num_outfits}개의 코디를 추천해주세요."
        )

        return "\n".join(lines)

    @staticmethod
    def _build_candidates_map(
        results: list[ProductSearchResult],
    ) -> dict[str, ProductCandidate]:
        """상품 후보를 product_id 기준 맵으로 변환"""
        candidates_map: dict[str, ProductCandidate] = {}
        for result in results:
            for candidate in result.candidates:
                candidates_map[candidate.product_id] = (
                    candidate
                )
        return candidates_map

    def _parse_response(
        self,
        response: dict,
        candidates_map: dict[str, ProductCandidate],
    ) -> ShopSearchResponse:

        content = response["choices"][0]["message"][
            "content"
        ]
        data = self._extract_json(content)

        outfits = []
        for outfit_data in data.get("outfits", []):
            items: list[ShopProduct] = []
            for item_data in outfit_data.get("items", []):
                product_id = str(item_data["product_id"])

                if product_id in candidates_map:
                    candidate = candidates_map[product_id]
                    items.append(
                        ShopProduct(
                            product_id=product_id,
                            title=candidate.title,
                            brand=candidate.brand,
                            price=candidate.price,
                            image_url=candidate.image_url,
                            link=candidate.link,
                            source=candidate.source,
                            category=candidate.category,
                        )
                    )
                else:
                    logger.warning(
                        f"LLM returned invalid "
                        f"product_id: {product_id}"
                    )

            if items:
                outfits.append(
                    ShopOutfit(
                        outfit_id=str(uuid.uuid4()),
                        items=items,
                    )
                )

        return ShopSearchResponse(
            query_summary=data.get(
                "query_summary", "쇼핑 코디 추천"
            ),
            outfits=outfits,
        )

    @staticmethod
    def _extract_json(content: str) -> dict:

        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [
                line
                for line in lines
                if not line.startswith("```")
            ]
            content = "\n".join(lines)
        return json.loads(content)

    @staticmethod
    def _empty_response(
        parsed: ShopParsedQuery,
    ) -> ShopSearchResponse:

        return ShopSearchResponse(
            query_summary="검색 결과가 없습니다",
            outfits=[],
        )
