import logging

from app.outfit.llm_client import OpenAIClient
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
        self.query_parser = query_parser or QueryParser(llm_client=OpenAIClient())
        self.search_builder = search_builder or SearchQueryBuilder()
        self.repository = repository or ClothingRepository()
        self.composer = composer or OutfitComposer()

    async def recommend(self, request: OutfitRequest) -> OutfitResponse:
        logger.info(f"Processing outfit request for user: {request.user_id}")

        parsed = await self.query_parser.parse(request.query)
        logger.info(f"Parsed query: occasion={parsed.occasion}, style={parsed.style}")

        search_queries = self.search_builder.build(parsed)
        logger.info(f"Generated {len(search_queries)} search queries")

        search_results = await self.repository.search_multiple(
            user_id=request.user_id,
            queries=search_queries,
        )

        total_candidates = sum(len(r.candidates) for r in search_results)
        logger.info(f"Found {total_candidates} total candidates")

        response = await self.composer.compose(
            parsed_query=parsed,
            search_results=search_results,
        )

        logger.info(f"Generated {len(response.outfits)} outfit recommendations")

        return response
