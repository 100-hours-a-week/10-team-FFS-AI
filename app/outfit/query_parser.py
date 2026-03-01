import logging

from langfuse.decorators import observe

from app.common.llm_schemas import OutfitQueryLLMResponse
from app.common.metrics import measure_time
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
   - category: 카테고리 (반드시 다음 중 하나로 매핑: TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC) (예: 코트->TOP, 바지->BOTTOM)
   - color: 색상
   - style: 스타일 (오버핏 등)
   - description: 기타 설명
6. target_category: 찾고 있는 아이템 카테고리 (반드시 다음 중 하나로 매핑: TOP, BOTTOM, DRESS, SHOES, ACCESSORY, ETC). 전체 코디 요청이면 null
7. constraints: 추가 제약사항 배열 (밝은 색으로, 편한 신발 등)"""


class QueryParser:
    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    @observe(name="query_parser.parse")
    @measure_time("query_parser")
    async def parse(
        self, query: str, trace_id: str | None = None, user_id: int | None = None
    ) -> ParsedQuery:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        log_context = f"trace_id={trace_id}" if trace_id else ""
        if user_id is not None:
            log_context += (
                f" user_id={user_id}" if log_context else f"user_id={user_id}"
            )

        logger.info(f'Parsing query | {log_context} query="{query}"')

        try:
            response: OutfitQueryLLMResponse = await self.llm_client.chat_completion(
                messages=messages,
                response_format=OutfitQueryLLMResponse,
                temperature=0.0,
                max_tokens=500,
            )
            return self._to_parsed_query(response)

        except LLMError:
            raise

        except Exception as e:
            logger.exception(
                f"Unexpected error parsing query | {log_context} error={e}"
            )
            raise ParseError(f"Unexpected parsing error: {e}") from e

    @staticmethod
    def _to_parsed_query(response: OutfitQueryLLMResponse) -> ParsedQuery:
        reference_item = None
        if response.reference_item:
            reference_item = ReferenceItem(
                category=response.reference_item.category,
                color=response.reference_item.color,
                style=response.reference_item.style,
                description=response.reference_item.description,
            )
        return ParsedQuery(
            occasion=response.occasion,
            style=response.style,
            season=response.season,
            formality=response.formality,
            reference_item=reference_item,
            target_category=response.target_category,
            constraints=response.constraints,
        )
