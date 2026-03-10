import logging

from langfuse import observe

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
   - category: 카테고리 (반드시 다음 중 하나로 매핑: TOP, BOTTOM, OUTER, DRESS, SHOES, ACCESSORY, ETC)
     (예: 코트→OUTER, 청바지→BOTTOM, 맨투맨→TOP)
   - color: 색상
   - style: 스타일 (오버핏 등)
   - description: 기타 설명
6. target_category: 찾고 있는 아이템 카테고리 (반드시 다음 중 하나로 매핑: TOP, BOTTOM, OUTER, DRESS, SHOES, ACCESSORY, ETC).
   전체 코디 요청이면 null
7. constraints: 추가 제약사항 배열 (밝은 색으로, 편한 신발 등)
8. sub_categories: 카테고리별 세부 유형 (사용자가 특정 아이템 유형을 언급하면 채울 것)
   형식: {"카테고리": ["세부유형", ...]}
   사용자가 명확히 언급했을 때만 포함. 불분명하면 null
   가능한 값:
   - TOP: 반소매_티셔츠, 긴소매_티셔츠, 셔츠_블라우스, 맨투맨_스웨트, 니트_스웨터
   - BOTTOM: 데님_팬츠, 슬랙스_트라우저, 트레이닝_조거, 숏츠, 스커트, 레깅스
   - OUTER: 가디건_니트아우터, 집업_후드아우터, 자켓, 코트, 패딩, 블레이저_수트자켓
   - SHOES: 스니커즈, 로퍼_단화, 구두_힐, 부츠, 샌들_슬리퍼
   - DRESS: 원피스, 점프슈트
   - ACCESSORY: 모자, 스카프_넥, 주얼리, 벨트, 양말_레그웨어, 백팩, 크로스_숄더백, 클러치_파우치, 웨이스트백
   (예: "맨투맨이랑 청바지" → {"TOP": ["맨투맨_스웨트"], "BOTTOM": ["데님_팬츠"]})"""


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
            sub_categories=response.sub_categories,
        )
