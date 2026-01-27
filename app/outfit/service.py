import logging

from app.outfit.outfit_composer import OutfitComposer
from app.outfit.query_parser import QueryParser
from app.outfit.repository import ClothingRepository
from app.outfit.schemas import OutfitRequest, OutfitResponse
from app.outfit.search_query_builder import SearchQueryBuilder

logger = logging.getLogger(__name__)


class OutfitService:
    def __init__(
        self,
        query_parser: QueryParser | None = None,
        search_builder: SearchQueryBuilder | None = None,
        repository: ClothingRepository | None = None,
        composer: OutfitComposer | None = None,
    ) -> None:
        self._query_parser = query_parser
        self._search_builder = search_builder or SearchQueryBuilder()
        self._repository = repository or ClothingRepository()
        self._composer = composer or OutfitComposer()

    def _get_query_parser(self) -> QueryParser:
        if self._query_parser is None:
            from app.outfit.llm_client import OpenAIClient

            self._query_parser = QueryParser(llm_client=OpenAIClient())
        return self._query_parser

    async def recommend(self, request: OutfitRequest) -> OutfitResponse:
        logger.info(f"Processing outfit request for user: {request.user_id}")

        try:
            parsed = await self._get_query_parser().parse(request.query)
            logger.info(f"Parsed query: occasion={parsed.occasion}, style={parsed.style}")
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            raise

        try:
            search_queries = self._get_search_builder().build(parsed)
            logger.info(f"Generated {len(search_queries)} search queries")
        except Exception as e:
            logger.error(f"Search query building failed: {e}")
            raise

        try:
            search_results = await self._get_repository().search_multiple(
                user_id=request.user_id,
                queries=search_queries,
            )
            total_candidates = sum(len(r.candidates) for r in search_results)
            logger.info(f"Found {total_candidates} total candidates")
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            raise

        try:
            response = await self._get_composer().compose(
                parsed_query=parsed,
                search_results=search_results,
            )
            logger.info(f"Generated {len(response.outfits)} outfit recommendations")
        except Exception as e:
            logger.error(f"Outfit composition failed: {e}")
            raise

        return response